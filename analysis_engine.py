"""
Analysis Engine Module for Real-Time Market Analysis

This module processes raw data streams and calculates key indicators in real-time:
- CVD and Spot/Perpetual Flow analysis
- Kalman Filter for signal smoothing
- GARCH(1,1) model for conditional volatility
- Hidden Markov Model for market regime identification
- Hurst Exponent for trend persistence
- Ichimoku Cloud indicators
"""

import math
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from collections import deque
from dataclasses import dataclass
import time

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class MarketIndicators:
    """Data class to hold calculated market indicators"""
    timestamp: float
    symbol: str
    cvd: float
    cvd_smoothed: float
    volatility: float
    regime: str
    hurst_exponent: float
    ichimoku_signal: str
    trend_strength: float


class KalmanFilter:
    """
    Kalman Filter implementation for signal smoothing
    """
    
    def __init__(self, process_variance: float = 1e-3, measurement_variance: float = 0.1):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        
        # Initial state
        self.state_estimate = 0.0
        self.error_covariance = 1.0
        
    def update(self, measurement: float) -> float:
        """Update filter with new measurement and return smoothed value"""
        # Prediction step
        predicted_state = self.state_estimate
        predicted_covariance = self.error_covariance + self.process_variance
        
        # Update step
        kalman_gain = predicted_covariance / (predicted_covariance + self.measurement_variance)
        self.state_estimate = predicted_state + kalman_gain * (measurement - predicted_state)
        self.error_covariance = (1 - kalman_gain) * predicted_covariance
        
        return self.state_estimate


class SimpleGARCH:
    """
    Simplified GARCH(1,1) model for conditional volatility estimation
    """
    
    def __init__(self, alpha: float = 0.1, beta: float = 0.85, omega: float = 0.01):
        self.alpha = alpha  # ARCH coefficient
        self.beta = beta    # GARCH coefficient
        self.omega = omega  # Constant term
        
        self.returns = deque(maxlen=100)
        self.conditional_variance = omega / (1 - alpha - beta)
        
    def update(self, return_value: float) -> float:
        """Update GARCH model with new return and calculate conditional volatility"""
        self.returns.append(return_value)
        
        if len(self.returns) >= 2:
            # GARCH(1,1) formula: σ²(t) = ω + α*ε²(t-1) + β*σ²(t-1)
            last_return_squared = self.returns[-2] ** 2
            self.conditional_variance = (self.omega + 
                                       self.alpha * last_return_squared + 
                                       self.beta * self.conditional_variance)
        
        return math.sqrt(self.conditional_variance)


class MarketRegimeDetector:
    """
    Simple Hidden Markov Model-like regime detection
    """
    
    def __init__(self, volatility_threshold: float = 0.02):
        self.volatility_threshold = volatility_threshold
        self.current_regime = "low_volatility"
        self.regime_history = deque(maxlen=50)
        
    def detect_regime(self, volatility: float, trend_strength: float) -> str:
        """Detect current market regime based on volatility and trend strength"""
        if volatility > self.volatility_threshold:
            if trend_strength > 0.6:
                regime = "high_volatility_trending"
            else:
                regime = "high_volatility_choppy"
        else:
            if trend_strength > 0.4:
                regime = "low_volatility_trending"
            else:
                regime = "low_volatility_ranging"
                
        self.regime_history.append(regime)
        self.current_regime = regime
        return regime


class HurstExponentCalculator:
    """
    Rolling Hurst Exponent calculation for trend persistence measurement
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.price_data = deque(maxlen=window_size)
        
    def calculate_hurst(self, prices: List[float]) -> float:
        """Calculate Hurst exponent using R/S analysis"""
        if len(prices) < 10:
            return 0.5  # Random walk default
            
        try:
            n = len(prices)
            
            # Calculate log returns
            log_returns = [math.log(prices[i] / prices[i-1]) for i in range(1, n)]
            
            # Calculate mean return
            mean_return = sum(log_returns) / len(log_returns)
            
            # Calculate cumulative deviations
            cumulative_deviations = []
            cumsum = 0
            for ret in log_returns:
                cumsum += ret - mean_return
                cumulative_deviations.append(cumsum)
            
            # Calculate range
            range_val = max(cumulative_deviations) - min(cumulative_deviations)
            
            # Calculate standard deviation
            variance = sum((ret - mean_return) ** 2 for ret in log_returns) / len(log_returns)
            std_dev = math.sqrt(variance)
            
            if std_dev == 0:
                return 0.5
                
            # R/S ratio
            rs_ratio = range_val / std_dev
            
            # Hurst exponent
            if rs_ratio > 0:
                hurst = math.log(rs_ratio) / math.log(n)
                return max(0.0, min(1.0, hurst))  # Clamp between 0 and 1
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Error calculating Hurst exponent: {e}")
            return 0.5
            
    def update(self, price: float) -> float:
        """Update with new price and return current Hurst exponent"""
        self.price_data.append(price)
        
        if len(self.price_data) >= 10:
            return self.calculate_hurst(list(self.price_data))
        else:
            return 0.5


class IchimokuCalculator:
    """
    Ichimoku Cloud indicators calculation
    """
    
    def __init__(self):
        self.tenkan_period = 9
        self.kijun_period = 26
        self.senkou_b_period = 52
        
        self.high_prices = deque(maxlen=self.senkou_b_period)
        self.low_prices = deque(maxlen=self.senkou_b_period)
        self.close_prices = deque(maxlen=self.senkou_b_period)
        
    def update(self, high: float, low: float, close: float) -> Dict[str, float]:
        """Update Ichimoku indicators with new OHLC data"""
        self.high_prices.append(high)
        self.low_prices.append(low)
        self.close_prices.append(close)
        
        indicators = {}
        
        # Tenkan-sen (Conversion Line)
        if len(self.high_prices) >= self.tenkan_period:
            tenkan_high = max(list(self.high_prices)[-self.tenkan_period:])
            tenkan_low = min(list(self.low_prices)[-self.tenkan_period:])
            indicators['tenkan_sen'] = (tenkan_high + tenkan_low) / 2
            
        # Kijun-sen (Base Line)
        if len(self.high_prices) >= self.kijun_period:
            kijun_high = max(list(self.high_prices)[-self.kijun_period:])
            kijun_low = min(list(self.low_prices)[-self.kijun_period:])
            indicators['kijun_sen'] = (kijun_high + kijun_low) / 2
            
        # Senkou Span A (Leading Span A)
        if 'tenkan_sen' in indicators and 'kijun_sen' in indicators:
            indicators['senkou_span_a'] = (indicators['tenkan_sen'] + indicators['kijun_sen']) / 2
            
        # Senkou Span B (Leading Span B)
        if len(self.high_prices) >= self.senkou_b_period:
            senkou_high = max(list(self.high_prices))
            senkou_low = min(list(self.low_prices))
            indicators['senkou_span_b'] = (senkou_high + senkou_low) / 2
            
        # Generate signal
        current_price = close
        signal = "neutral"
        
        if 'senkou_span_a' in indicators and 'senkou_span_b' in indicators:
            cloud_top = max(indicators['senkou_span_a'], indicators['senkou_span_b'])
            cloud_bottom = min(indicators['senkou_span_a'], indicators['senkou_span_b'])
            
            if current_price > cloud_top:
                signal = "bullish"
            elif current_price < cloud_bottom:
                signal = "bearish"
            else:
                signal = "inside_cloud"
                
        indicators['signal'] = signal
        return indicators


class AnalysisEngine:
    """
    Main analysis engine that coordinates all mathematical models
    """
    
    def __init__(self):
        self.kalman_filter = KalmanFilter()
        self.garch_model = SimpleGARCH()
        self.regime_detector = MarketRegimeDetector()
        self.hurst_calculator = HurstExponentCalculator()
        self.ichimoku_calculator = IchimokuCalculator()
        
        # Data storage
        self.price_history = deque(maxlen=1000)
        self.cvd_history = deque(maxlen=1000)
        self.volume_history = deque(maxlen=1000)
        
        # Current indicators
        self.current_indicators = None
        
    def update_with_trade_data(self, trade_data: Dict) -> MarketIndicators:
        """Update all models with new trade data"""
        try:
            timestamp = trade_data.get('timestamp', time.time() * 1000)
            symbol = trade_data.get('symbol', 'UNKNOWN')
            price = trade_data.get('price', 0.0)
            cvd = trade_data.get('cvd', 0.0)
            quantity = trade_data.get('quantity', 0.0)
            
            # Store historical data
            self.price_history.append(price)
            self.cvd_history.append(cvd)
            self.volume_history.append(quantity)
            
            # Calculate price return for GARCH model
            price_return = 0.0
            if len(self.price_history) >= 2:
                price_return = (price - self.price_history[-2]) / self.price_history[-2]
            
            # Update all models
            cvd_smoothed = self.kalman_filter.update(cvd)
            volatility = self.garch_model.update(price_return)
            hurst_exponent = self.hurst_calculator.update(price)
            
            # Calculate trend strength (simplified)
            trend_strength = self.calculate_trend_strength()
            
            # Detect market regime
            regime = self.regime_detector.detect_regime(volatility, trend_strength)
            
            # Update Ichimoku (using price as high/low/close for simplicity)
            ichimoku_data = self.ichimoku_calculator.update(price, price, price)
            ichimoku_signal = ichimoku_data.get('signal', 'neutral')
            
            # Create indicators object
            indicators = MarketIndicators(
                timestamp=timestamp,
                symbol=symbol,
                cvd=cvd,
                cvd_smoothed=cvd_smoothed,
                volatility=volatility,
                regime=regime,
                hurst_exponent=hurst_exponent,
                ichimoku_signal=ichimoku_signal,
                trend_strength=trend_strength
            )
            
            self.current_indicators = indicators
            
            # Log significant changes
            if len(self.price_history) % 50 == 0:  # Log every 50 updates
                logger.info(f"Analysis Update - {symbol}: "
                          f"Price: ${price:.2f}, CVD: {cvd:.2f}, "
                          f"Volatility: {volatility:.4f}, Regime: {regime}, "
                          f"Hurst: {hurst_exponent:.3f}, Ichimoku: {ichimoku_signal}")
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error in analysis engine update: {e}")
            return None
            
    def calculate_trend_strength(self) -> float:
        """Calculate trend strength based on price momentum"""
        if len(self.price_history) < 20:
            return 0.0
            
        try:
            # Simple trend strength based on linear regression slope
            prices = list(self.price_history)[-20:]  # Last 20 prices
            n = len(prices)
            x = list(range(n))
            
            # Calculate linear regression slope
            sum_x = sum(x)
            sum_y = sum(prices)
            sum_xy = sum(x[i] * prices[i] for i in range(n))
            sum_x_squared = sum(xi ** 2 for xi in x)
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)
            
            # Normalize slope to 0-1 range
            max_price = max(prices)
            min_price = min(prices)
            price_range = max_price - min_price
            
            if price_range == 0:
                return 0.0
                
            normalized_slope = abs(slope) / (price_range / n)
            return min(1.0, normalized_slope)
            
        except Exception as e:
            logger.error(f"Error calculating trend strength: {e}")
            return 0.0
            
    def get_current_indicators(self) -> Optional[MarketIndicators]:
        """Get the most recent market indicators"""
        return self.current_indicators
        
    def get_cvd_divergence(self, lookback_periods: int = 50) -> float:
        """Calculate CVD divergence (spot vs perpetual simulation)"""
        if len(self.cvd_history) < lookback_periods:
            return 0.0
            
        try:
            # Simulate spot vs perpetual divergence
            recent_cvd = list(self.cvd_history)[-lookback_periods:]
            spot_cvd = sum(recent_cvd[:len(recent_cvd)//2])
            perp_cvd = sum(recent_cvd[len(recent_cvd)//2:])
            
            if abs(perp_cvd) < 0.001:  # Avoid division by zero
                return 0.0
                
            divergence = (spot_cvd - perp_cvd) / abs(perp_cvd)
            return divergence
            
        except Exception as e:
            logger.error(f"Error calculating CVD divergence: {e}")
            return 0.0


# Example usage
if __name__ == "__main__":
    # Test the analysis engine
    engine = AnalysisEngine()
    
    # Simulate some trade data
    import random
    base_price = 50000.0
    cumulative_cvd = 0.0
    
    for i in range(100):
        # Simulate price movement
        price_change = random.gauss(0, 100)
        price = base_price + price_change
        base_price = price
        
        # Simulate CVD
        volume_delta = random.gauss(0, 10)
        cumulative_cvd += volume_delta
        
        # Create mock trade data
        trade_data = {
            'timestamp': time.time() * 1000 + i * 1000,
            'symbol': 'BTCUSDT',
            'price': price,
            'quantity': abs(volume_delta),
            'cvd': cumulative_cvd,
            'side': 'buy' if volume_delta > 0 else 'sell'
        }
        
        # Update analysis engine
        indicators = engine.update_with_trade_data(trade_data)
        
        if indicators and i % 20 == 0:
            print(f"Indicators: Price=${indicators.cvd:.2f}, "
                  f"Regime={indicators.regime}, "
                  f"Hurst={indicators.hurst_exponent:.3f}")
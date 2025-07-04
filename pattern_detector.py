"""
Pattern Detection Module for Real-Time Market Analysis

This module analyzes output from the analysis engine to identify specific patterns:
- Bull/Bear trap detection via CVD divergence
- Liquidation cascade alerts
- Market regime shift detection
- Volume anomaly detection
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
from enum import Enum
import time
import math

# Import our analysis engine types
from analysis_engine import MarketIndicators

# Setup logging
logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class MarketAlert:
    """Data class for market alerts"""
    timestamp: float
    alert_type: str
    severity: AlertSeverity
    symbol: str
    message: str
    confidence: float  # 0-1 scale
    data: Dict  # Additional alert-specific data


class BullBearTrapDetector:
    """
    Detector for bull and bear traps using CVD divergence analysis
    """
    
    def __init__(self, divergence_threshold: float = 0.15, confidence_window: int = 20):
        self.divergence_threshold = divergence_threshold
        self.confidence_window = confidence_window
        
        # Historical data for pattern detection
        self.price_history = deque(maxlen=100)
        self.cvd_history = deque(maxlen=100)
        self.volume_history = deque(maxlen=100)
        
        # Pattern state tracking
        self.last_alert_time = 0
        self.alert_cooldown = 300  # 5 minutes between similar alerts
        
    def detect_trap(self, indicators: MarketIndicators, cvd_divergence: float) -> Optional[MarketAlert]:
        """Detect bull/bear traps based on price action and CVD divergence"""
        try:
            # Store current data
            self.price_history.append(indicators.cvd)  # Using CVD as price proxy for demo
            self.cvd_history.append(indicators.cvd)
            
            # Need sufficient data
            if len(self.price_history) < self.confidence_window:
                return None
                
            current_time = time.time()
            
            # Check alert cooldown
            if current_time - self.last_alert_time < self.alert_cooldown:
                return None
                
            # Calculate recent price momentum
            recent_prices = list(self.price_history)[-self.confidence_window:]
            price_change = (recent_prices[-1] - recent_prices[0]) / abs(recent_prices[0]) if recent_prices[0] != 0 else 0
            
            # Detect bull trap: Price rising but CVD diverging negatively
            if price_change > 0.02 and cvd_divergence < -self.divergence_threshold:
                confidence = min(0.9, abs(cvd_divergence) + abs(price_change))
                
                alert = MarketAlert(
                    timestamp=indicators.timestamp,
                    alert_type="BULL_TRAP",
                    severity=AlertSeverity.HIGH if confidence > 0.7 else AlertSeverity.MEDIUM,
                    symbol=indicators.symbol,
                    message=f"Potential Bull Trap Detected - Price rising {price_change:.2%} but CVD diverging {cvd_divergence:.2%}",
                    confidence=confidence,
                    data={
                        'price_change': price_change,
                        'cvd_divergence': cvd_divergence,
                        'volatility': indicators.volatility,
                        'regime': indicators.regime
                    }
                )
                
                self.last_alert_time = current_time
                return alert
                
            # Detect bear trap: Price falling but CVD diverging positively
            elif price_change < -0.02 and cvd_divergence > self.divergence_threshold:
                confidence = min(0.9, abs(cvd_divergence) + abs(price_change))
                
                alert = MarketAlert(
                    timestamp=indicators.timestamp,
                    alert_type="BEAR_TRAP",
                    severity=AlertSeverity.HIGH if confidence > 0.7 else AlertSeverity.MEDIUM,
                    symbol=indicators.symbol,
                    message=f"Potential Bear Trap Detected - Price falling {price_change:.2%} but CVD diverging {cvd_divergence:.2%}",
                    confidence=confidence,
                    data={
                        'price_change': price_change,
                        'cvd_divergence': cvd_divergence,
                        'volatility': indicators.volatility,
                        'regime': indicators.regime
                    }
                )
                
                self.last_alert_time = current_time
                return alert
                
            return None
            
        except Exception as e:
            logger.error(f"Error in bull/bear trap detection: {e}")
            return None


class LiquidationCascadeDetector:
    """
    Detector for potential liquidation cascades
    """
    
    def __init__(self, volume_spike_threshold: float = 3.0, volatility_threshold: float = 0.05):
        self.volume_spike_threshold = volume_spike_threshold
        self.volatility_threshold = volatility_threshold
        
        self.volume_baseline = deque(maxlen=50)
        self.liquidation_levels = []  # Simulated liquidation levels
        self.last_alert_time = 0
        
    def update_liquidation_levels(self, liquidation_data: List[Dict]):
        """Update known liquidation levels from data feed"""
        # In a real implementation, this would process actual liquidation data
        # For now, we'll simulate some liquidation levels
        self.liquidation_levels = [
            {'price': 50000, 'amount': 100},
            {'price': 49500, 'amount': 150},
            {'price': 51000, 'amount': 80}
        ]
        
    def detect_cascade(self, indicators: MarketIndicators, current_price: float) -> Optional[MarketAlert]:
        """Detect potential liquidation cascades"""
        try:
            current_time = time.time()
            
            # Check alert cooldown
            if current_time - self.last_alert_time < 180:  # 3 minute cooldown
                return None
                
            # Check if volatility is elevated
            if indicators.volatility < self.volatility_threshold:
                return None
                
            # Check if price is approaching liquidation levels
            approaching_liquidations = []
            price_tolerance = current_price * 0.02  # 2% tolerance
            
            for liq_level in self.liquidation_levels:
                distance = abs(current_price - liq_level['price'])
                if distance <= price_tolerance:
                    approaching_liquidations.append(liq_level)
                    
            if approaching_liquidations:
                total_liq_amount = sum(liq['amount'] for liq in approaching_liquidations)
                
                # Calculate risk level
                if total_liq_amount > 200:
                    severity = AlertSeverity.CRITICAL
                elif total_liq_amount > 100:
                    severity = AlertSeverity.HIGH
                else:
                    severity = AlertSeverity.MEDIUM
                    
                confidence = min(0.9, (total_liq_amount / 300) + (indicators.volatility / 0.1))
                
                alert = MarketAlert(
                    timestamp=indicators.timestamp,
                    alert_type="LIQUIDATION_CASCADE",
                    severity=severity,
                    symbol=indicators.symbol,
                    message=f"Liquidation Cascade Alert - ${total_liq_amount:.0f}M approaching at ${current_price:.0f}",
                    confidence=confidence,
                    data={
                        'liquidation_amount': total_liq_amount,
                        'liquidation_levels': approaching_liquidations,
                        'current_volatility': indicators.volatility,
                        'regime': indicators.regime
                    }
                )
                
                self.last_alert_time = current_time
                return alert
                
            return None
            
        except Exception as e:
            logger.error(f"Error in liquidation cascade detection: {e}")
            return None


class RegimeShiftDetector:
    """
    Detector for market regime shifts using HMM and volatility analysis
    """
    
    def __init__(self):
        self.regime_history = deque(maxlen=30)
        self.volatility_history = deque(maxlen=50)
        self.last_regime = None
        self.last_alert_time = 0
        
    def detect_shift(self, indicators: MarketIndicators) -> Optional[MarketAlert]:
        """Detect significant regime shifts"""
        try:
            # Store current regime and volatility
            self.regime_history.append(indicators.regime)
            self.volatility_history.append(indicators.volatility)
            
            current_time = time.time()
            
            # Check if regime has changed
            if self.last_regime and self.last_regime != indicators.regime:
                # Check if this is a significant shift
                is_significant_shift = self._is_significant_shift(self.last_regime, indicators.regime)
                
                if is_significant_shift and (current_time - self.last_alert_time > 600):  # 10 minute cooldown
                    
                    # Calculate volatility trend
                    vol_trend = self._calculate_volatility_trend()
                    
                    severity = self._assess_shift_severity(self.last_regime, indicators.regime, vol_trend)
                    
                    alert = MarketAlert(
                        timestamp=indicators.timestamp,
                        alert_type="REGIME_SHIFT",
                        severity=severity,
                        symbol=indicators.symbol,
                        message=f"Market Regime Shift: {self.last_regime} → {indicators.regime}",
                        confidence=0.8,
                        data={
                            'previous_regime': self.last_regime,
                            'new_regime': indicators.regime,
                            'volatility_trend': vol_trend,
                            'hurst_exponent': indicators.hurst_exponent,
                            'current_volatility': indicators.volatility
                        }
                    )
                    
                    self.last_alert_time = current_time
                    self.last_regime = indicators.regime
                    return alert
                    
            self.last_regime = indicators.regime
            return None
            
        except Exception as e:
            logger.error(f"Error in regime shift detection: {e}")
            return None
            
    def _is_significant_shift(self, old_regime: str, new_regime: str) -> bool:
        """Determine if a regime shift is significant enough to alert"""
        significant_shifts = [
            ("low_volatility", "high_volatility_trending"),
            ("low_volatility", "high_volatility_choppy"),
            ("high_volatility_choppy", "high_volatility_trending"),
            ("low_volatility_trending", "high_volatility_choppy")
        ]
        
        for old, new in significant_shifts:
            if old in old_regime and new in new_regime:
                return True
        return False
        
    def _calculate_volatility_trend(self) -> str:
        """Calculate if volatility is increasing or decreasing"""
        if len(self.volatility_history) < 10:
            return "unknown"
            
        recent_vol = list(self.volatility_history)[-5:]
        older_vol = list(self.volatility_history)[-10:-5]
        
        avg_recent = sum(recent_vol) / len(recent_vol)
        avg_older = sum(older_vol) / len(older_vol)
        
        if avg_recent > avg_older * 1.2:
            return "increasing"
        elif avg_recent < avg_older * 0.8:
            return "decreasing"
        else:
            return "stable"
            
    def _assess_shift_severity(self, old_regime: str, new_regime: str, vol_trend: str) -> AlertSeverity:
        """Assess the severity of a regime shift"""
        if "high_volatility" in new_regime and vol_trend == "increasing":
            return AlertSeverity.HIGH
        elif "high_volatility_choppy" in new_regime:
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW


class VolumeAnomalyDetector:
    """
    Detector for unusual volume patterns
    """
    
    def __init__(self, anomaly_threshold: float = 3.0):
        self.anomaly_threshold = anomaly_threshold
        self.volume_history = deque(maxlen=100)
        self.last_alert_time = 0
        
    def detect_anomaly(self, volume: float, indicators: MarketIndicators) -> Optional[MarketAlert]:
        """Detect volume anomalies using statistical analysis"""
        try:
            self.volume_history.append(volume)
            
            if len(self.volume_history) < 20:
                return None
                
            current_time = time.time()
            
            # Check cooldown
            if current_time - self.last_alert_time < 120:  # 2 minute cooldown
                return None
                
            # Calculate volume statistics
            volumes = list(self.volume_history)
            mean_volume = sum(volumes) / len(volumes)
            variance = sum((v - mean_volume) ** 2 for v in volumes) / len(volumes)
            std_dev = math.sqrt(variance)
            
            if std_dev == 0:
                return None
                
            # Calculate z-score for current volume
            z_score = (volume - mean_volume) / std_dev
            
            if abs(z_score) > self.anomaly_threshold:
                anomaly_type = "HIGH_VOLUME" if z_score > 0 else "LOW_VOLUME"
                
                severity = AlertSeverity.HIGH if abs(z_score) > 4 else AlertSeverity.MEDIUM
                confidence = min(0.95, abs(z_score) / 5)
                
                alert = MarketAlert(
                    timestamp=indicators.timestamp,
                    alert_type=f"VOLUME_ANOMALY_{anomaly_type}",
                    severity=severity,
                    symbol=indicators.symbol,
                    message=f"Volume Anomaly Detected - {anomaly_type}: {volume:.2f} (Z-score: {z_score:.2f})",
                    confidence=confidence,
                    data={
                        'volume': volume,
                        'mean_volume': mean_volume,
                        'z_score': z_score,
                        'regime': indicators.regime
                    }
                )
                
                self.last_alert_time = current_time
                return alert
                
            return None
            
        except Exception as e:
            logger.error(f"Error in volume anomaly detection: {e}")
            return None


class PatternDetector:
    """
    Main pattern detection coordinator
    """
    
    def __init__(self):
        self.bull_bear_detector = BullBearTrapDetector()
        self.liquidation_detector = LiquidationCascadeDetector()
        self.regime_detector = RegimeShiftDetector()
        self.volume_detector = VolumeAnomalyDetector()
        
        self.alert_history = deque(maxlen=1000)
        
    def analyze_patterns(self, indicators: MarketIndicators, cvd_divergence: float, 
                        current_price: float, volume: float) -> List[MarketAlert]:
        """Run all pattern detection algorithms and return any alerts"""
        alerts = []
        
        try:
            # Bull/Bear trap detection
            trap_alert = self.bull_bear_detector.detect_trap(indicators, cvd_divergence)
            if trap_alert:
                alerts.append(trap_alert)
                
            # Liquidation cascade detection
            cascade_alert = self.liquidation_detector.detect_cascade(indicators, current_price)
            if cascade_alert:
                alerts.append(cascade_alert)
                
            # Regime shift detection
            regime_alert = self.regime_detector.detect_shift(indicators)
            if regime_alert:
                alerts.append(regime_alert)
                
            # Volume anomaly detection
            volume_alert = self.volume_detector.detect_anomaly(volume, indicators)
            if volume_alert:
                alerts.append(volume_alert)
                
            # Store alerts in history
            for alert in alerts:
                self.alert_history.append(alert)
                logger.info(f"ALERT: {alert.alert_type} - {alert.message} (Confidence: {alert.confidence:.2f})")
                
            return alerts
            
        except Exception as e:
            logger.error(f"Error in pattern analysis: {e}")
            return []
            
    def get_recent_alerts(self, minutes: int = 60) -> List[MarketAlert]:
        """Get alerts from the last N minutes"""
        cutoff_time = time.time() - (minutes * 60)
        return [alert for alert in self.alert_history if alert.timestamp > cutoff_time * 1000]
        
    def get_alert_summary(self, minutes: int = 60) -> Dict[str, int]:
        """Get summary of alert types in the last N minutes"""
        recent_alerts = self.get_recent_alerts(minutes)
        summary = {}
        
        for alert in recent_alerts:
            alert_type = alert.alert_type
            if alert_type not in summary:
                summary[alert_type] = 0
            summary[alert_type] += 1
            
        return summary


# Example usage
if __name__ == "__main__":
    # Test pattern detection
    from analysis_engine import MarketIndicators
    
    detector = PatternDetector()
    
    # Create mock indicators
    indicators = MarketIndicators(
        timestamp=time.time() * 1000,
        symbol="BTCUSDT",
        cvd=1000.0,
        cvd_smoothed=990.0,
        volatility=0.08,
        regime="high_volatility_trending",
        hurst_exponent=0.7,
        ichimoku_signal="bullish",
        trend_strength=0.8
    )
    
    # Test pattern detection
    alerts = detector.analyze_patterns(
        indicators=indicators,
        cvd_divergence=-0.2,  # Strong negative divergence
        current_price=50000,
        volume=150.0
    )
    
    for alert in alerts:
        print(f"Alert: {alert.alert_type} - {alert.message}")
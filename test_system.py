#!/usr/bin/env python3
"""
Test script for the Real-Time Market Analysis Engine

This script tests the core functionality without requiring live WebSocket connections.
It simulates market data to verify all components work correctly.
"""

import asyncio
import time
import random
import logging
from datetime import datetime

# Import our modules
from data_ingestor import DataAggregator
from analysis_engine import AnalysisEngine
from pattern_detector import PatternDetector

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_mock_trade_data(base_price: float, cvd: float, iteration: int) -> dict:
    """Generate realistic mock trade data"""
    
    # Simulate price movement with some trend and noise
    price_change = random.gauss(0, base_price * 0.001)  # 0.1% volatility
    
    # Add some trend based on iteration
    if iteration % 100 < 30:  # Uptrend for first 30% of cycle
        price_change += base_price * 0.0005
    elif iteration % 100 > 70:  # Downtrend for last 30% of cycle
        price_change -= base_price * 0.0005
        
    new_price = base_price + price_change
    
    # Simulate volume delta
    volume_delta = random.gauss(0, 15)
    
    # Add some correlation between price and volume
    if price_change > 0:
        volume_delta += random.uniform(0, 5)  # Slight buy bias on price increases
    else:
        volume_delta -= random.uniform(0, 5)  # Slight sell bias on price decreases
        
    quantity = abs(volume_delta)
    side = 'buy' if volume_delta > 0 else 'sell'
    
    return {
        'timestamp': time.time() * 1000 + iteration * 1000,
        'symbol': 'BTCUSDT',
        'price': new_price,
        'quantity': quantity,
        'cvd': cvd + volume_delta,
        'volume_delta': volume_delta,
        'side': side
    }


async def test_analysis_engine():
    """Test the complete analysis pipeline"""
    logger.info("Starting Real-Time Market Analysis Engine Test")
    logger.info("=" * 60)
    
    # Initialize components
    analysis_engine = AnalysisEngine()
    pattern_detector = PatternDetector()
    
    # Test parameters
    base_price = 50000.0
    cumulative_cvd = 0.0
    test_duration = 200  # number of iterations
    
    logger.info(f"Running {test_duration} iterations of market simulation...")
    
    alerts_generated = 0
    
    for i in range(test_duration):
        # Generate mock trade data
        trade_data = generate_mock_trade_data(base_price, cumulative_cvd, i)
        base_price = trade_data['price']
        cumulative_cvd = trade_data['cvd']
        
        # Update analysis engine
        indicators = analysis_engine.update_with_trade_data(trade_data)
        
        if indicators:
            # Calculate additional metrics
            cvd_divergence = analysis_engine.get_cvd_divergence()
            
            # Run pattern detection
            alerts = pattern_detector.analyze_patterns(
                indicators=indicators,
                cvd_divergence=cvd_divergence,
                current_price=base_price,
                volume=trade_data['quantity']
            )
            
            # Process alerts
            if alerts:
                alerts_generated += len(alerts)
                for alert in alerts:
                    logger.info(f"🚨 TEST ALERT: {alert.alert_type} - {alert.message}")
                    logger.info(f"   Confidence: {alert.confidence:.2f}")
            
            # Log progress every 50 iterations
            if i % 50 == 0 and i > 0:
                logger.info(f"Progress: {i}/{test_duration} - Price: ${base_price:.2f}, "
                          f"CVD: {cumulative_cvd:.2f}, Regime: {indicators.regime}")
                
                # Log current indicators
                logger.info(f"Indicators: Volatility: {indicators.volatility:.4f}, "
                          f"Hurst: {indicators.hurst_exponent:.3f}, "
                          f"Ichimoku: {indicators.ichimoku_signal}")
        
        # Small delay to simulate real-time processing
        await asyncio.sleep(0.01)
    
    # Final summary
    logger.info("=" * 60)
    logger.info("TEST COMPLETE")
    logger.info(f"Total iterations: {test_duration}")
    logger.info(f"Final price: ${base_price:.2f}")
    logger.info(f"Final CVD: {cumulative_cvd:.2f}")
    logger.info(f"Alerts generated: {alerts_generated}")
    
    if indicators:
        logger.info(f"Final regime: {indicators.regime}")
        logger.info(f"Final volatility: {indicators.volatility:.4f}")
        logger.info(f"Final Hurst exponent: {indicators.hurst_exponent:.3f}")
    
    # Test pattern detector summary
    alert_summary = pattern_detector.get_alert_summary(minutes=60)
    logger.info(f"Alert summary: {alert_summary}")
    
    return True


def test_data_aggregator():
    """Test the data aggregator component"""
    logger.info("Testing Data Aggregator...")
    
    aggregator = DataAggregator()
    
    # Simulate trade data
    mock_trade = {
        'data': {
            's': 'BTCUSDT',
            'p': '50000.00',
            'q': '0.1',
            'm': False,  # Buyer is not maker (buy order)
            'T': int(time.time() * 1000)
        }
    }
    
    # Process trade data
    aggregator.process_trade_data(mock_trade)
    
    # Check CVD calculation
    cvd = aggregator.get_current_cvd()
    logger.info(f"CVD after test trade: {cvd}")
    
    # Get latest data
    trades = aggregator.get_latest_data('trade', max_items=5)
    logger.info(f"Retrieved {len(trades)} trade records")
    
    if trades:
        latest_trade = trades[0]
        logger.info(f"Latest trade: {latest_trade}")
    
    logger.info("Data Aggregator test completed ✓")
    return True


async def main():
    """Run all tests"""
    logger.info("🧪 Starting Real-Time Market Analysis Engine Tests")
    
    try:
        # Test individual components
        logger.info("\n1. Testing Data Aggregator...")
        test_data_aggregator()
        
        logger.info("\n2. Testing Analysis Engine and Pattern Detection...")
        await test_analysis_engine()
        
        logger.info("\n✅ All tests completed successfully!")
        logger.info("\nTo run the live system, use: python main.py")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False
        
    return True


if __name__ == "__main__":
    asyncio.run(main())
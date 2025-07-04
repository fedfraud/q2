"""
Main Application for Real-Time Market Analysis Engine

This is the main orchestrator that coordinates all modules:
- Data ingestion from exchanges
- Real-time mathematical analysis
- Pattern detection and alerting
- Logging and optional dashboard integration
"""

import asyncio
import logging
import signal
import sys
import time
from datetime import datetime
from typing import Dict, List
import json
import os

# Import our modules
from data_ingestor import MarketDataIngestor, DataAggregator
from analysis_engine import AnalysisEngine, MarketIndicators
from pattern_detector import PatternDetector, MarketAlert, AlertSeverity

# Setup comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('market_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class MarketAnalysisEngine:
    """
    Main market analysis engine that coordinates all components
    """
    
    def __init__(self, symbol: str = "BTCUSDT", log_interval: int = 30):
        self.symbol = symbol
        self.log_interval = log_interval  # seconds between detailed logs
        
        # Initialize all components
        self.data_ingestor = MarketDataIngestor(symbol)
        self.analysis_engine = AnalysisEngine()
        self.pattern_detector = PatternDetector()
        
        # Runtime tracking
        self.running = False
        self.start_time = None
        self.trade_count = 0
        self.last_detailed_log = 0
        
        # Performance metrics
        self.performance_metrics = {
            'total_trades_processed': 0,
            'total_alerts_generated': 0,
            'analysis_updates': 0,
            'errors': 0
        }
        
        # Current market state
        self.current_price = 0.0
        self.current_indicators = None
        self.latest_alerts = []
        
    async def start_analysis(self):
        """Start the real-time market analysis engine"""
        logger.info(f"Starting Real-Time Market Analysis Engine for {self.symbol}")
        logger.info("=" * 80)
        
        self.running = True
        self.start_time = time.time()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        try:
            # Start data ingestion in the background
            ingestion_task = asyncio.create_task(self.data_ingestor.start_ingestion())
            
            # Start analysis loop
            analysis_task = asyncio.create_task(self._analysis_loop())
            
            # Wait for tasks to complete
            await asyncio.gather(ingestion_task, analysis_task, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"Critical error in analysis engine: {e}")
            self.performance_metrics['errors'] += 1
        finally:
            await self._shutdown()
            
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.running = False
        
    async def _analysis_loop(self):
        """Main analysis loop that processes data and generates insights"""
        logger.info("Starting analysis loop...")
        
        while self.running:
            try:
                await asyncio.sleep(1)  # Process every second
                
                # Get latest data from aggregator
                aggregator = self.data_ingestor.get_aggregator()
                latest_trades = aggregator.get_latest_data('trade', max_items=10)
                latest_orderbook = aggregator.get_latest_data('orderbook', max_items=1)
                
                # Process each new trade
                for trade_data in latest_trades:
                    await self._process_trade_data(trade_data, aggregator)
                    
                # Log detailed analysis periodically
                current_time = time.time()
                if current_time - self.last_detailed_log >= self.log_interval:
                    await self._log_detailed_analysis()
                    self.last_detailed_log = current_time
                    
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}")
                self.performance_metrics['errors'] += 1
                await asyncio.sleep(5)  # Wait before retrying
                
    async def _process_trade_data(self, trade_data: Dict, aggregator: DataAggregator):
        """Process individual trade data through the analysis pipeline"""
        try:
            self.trade_count += 1
            self.current_price = trade_data.get('price', self.current_price)
            
            # Update analysis engine with new trade data
            indicators = self.analysis_engine.update_with_trade_data(trade_data)
            
            if indicators:
                self.current_indicators = indicators
                self.performance_metrics['analysis_updates'] += 1
                
                # Calculate additional metrics for pattern detection
                cvd_divergence = self.analysis_engine.get_cvd_divergence()
                volume = trade_data.get('quantity', 0.0)
                
                # Run pattern detection
                alerts = self.pattern_detector.analyze_patterns(
                    indicators=indicators,
                    cvd_divergence=cvd_divergence,
                    current_price=self.current_price,
                    volume=volume
                )
                
                # Process any alerts
                if alerts:
                    await self._process_alerts(alerts)
                    
        except Exception as e:
            logger.error(f"Error processing trade data: {e}")
            self.performance_metrics['errors'] += 1
            
    async def _process_alerts(self, alerts: List[MarketAlert]):
        """Process and log market alerts"""
        for alert in alerts:
            self.performance_metrics['total_alerts_generated'] += 1
            self.latest_alerts.append(alert)
            
            # Keep only recent alerts
            cutoff_time = time.time() * 1000 - (3600 * 1000)  # 1 hour
            self.latest_alerts = [a for a in self.latest_alerts if a.timestamp > cutoff_time]
            
            # Log alert with appropriate severity
            alert_msg = f"🚨 {alert.severity.value} ALERT: {alert.alert_type} - {alert.message}"
            
            if alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
                logger.warning(alert_msg)
                # In a production system, this could trigger notifications
                await self._send_high_priority_alert(alert)
            else:
                logger.info(alert_msg)
                
            # Log alert details
            logger.info(f"   Confidence: {alert.confidence:.2f}, Symbol: {alert.symbol}")
            logger.info(f"   Data: {json.dumps(alert.data, indent=2)}")
            
    async def _send_high_priority_alert(self, alert: MarketAlert):
        """Send high priority alerts (placeholder for notification system)"""
        # In a real system, this would send alerts via:
        # - Email
        # - Slack/Discord webhooks
        # - SMS
        # - Push notifications
        # - Trading system integration
        
        alert_file = f"high_priority_alerts_{datetime.now().strftime('%Y%m%d')}.json"
        
        try:
            # Load existing alerts
            if os.path.exists(alert_file):
                with open(alert_file, 'r') as f:
                    existing_alerts = json.load(f)
            else:
                existing_alerts = []
                
            # Add new alert
            alert_dict = {
                'timestamp': alert.timestamp,
                'type': alert.alert_type,
                'severity': alert.severity.value,
                'symbol': alert.symbol,
                'message': alert.message,
                'confidence': alert.confidence,
                'data': alert.data
            }
            
            existing_alerts.append(alert_dict)
            
            # Save updated alerts
            with open(alert_file, 'w') as f:
                json.dump(existing_alerts, f, indent=2)
                
            logger.info(f"High priority alert saved to {alert_file}")
            
        except Exception as e:
            logger.error(f"Error saving high priority alert: {e}")
            
    async def _log_detailed_analysis(self):
        """Log detailed mathematical analysis similar to professional reviews"""
        if not self.current_indicators:
            return
            
        try:
            indicators = self.current_indicators
            runtime = time.time() - self.start_time
            
            # Get recent alerts summary
            alert_summary = self.pattern_detector.get_alert_summary(minutes=30)
            
            # Generate professional-style analysis log
            analysis_log = f"""
================================================================================
REAL-TIME MARKET ANALYSIS REPORT
================================================================================
Symbol: {indicators.symbol}
Timestamp: {datetime.fromtimestamp(indicators.timestamp/1000).strftime('%Y-%m-%d %H:%M:%S UTC')}
Runtime: {runtime/3600:.2f} hours
Current Price: ${self.current_price:,.2f}

MATHEMATICAL INDICATORS:
├─ Cumulative Volume Delta (CVD): {indicators.cvd:,.2f}
├─ CVD (Kalman Smoothed): {indicators.cvd_smoothed:,.2f}
├─ Conditional Volatility (GARCH): {indicators.volatility:.4f}
├─ Market Regime: {indicators.regime}
├─ Hurst Exponent: {indicators.hurst_exponent:.3f}
├─ Ichimoku Signal: {indicators.ichimoku_signal}
└─ Trend Strength: {indicators.trend_strength:.3f}

PATTERN ANALYSIS:
├─ CVD Divergence: {self.analysis_engine.get_cvd_divergence():.3f}
├─ Recent Alerts (30min): {len(self.pattern_detector.get_recent_alerts(30))}
└─ Alert Summary: {alert_summary}

PERFORMANCE METRICS:
├─ Trades Processed: {self.performance_metrics['total_trades_processed']:,}
├─ Analysis Updates: {self.performance_metrics['analysis_updates']:,}
├─ Alerts Generated: {self.performance_metrics['total_alerts_generated']:,}
└─ Errors: {self.performance_metrics['errors']:,}

MARKET INTERPRETATION:
"""
            
            # Add interpretation based on current conditions
            if indicators.volatility > 0.05:
                analysis_log += "⚠️  ELEVATED VOLATILITY - Market showing increased uncertainty\n"
                
            if indicators.hurst_exponent > 0.6:
                analysis_log += "📈 TRENDING MARKET - Strong directional persistence detected\n"
            elif indicators.hurst_exponent < 0.4:
                analysis_log += "🔄 MEAN REVERTING - Market showing reversal tendencies\n"
            else:
                analysis_log += "⚖️  RANDOM WALK - Market showing neutral persistence\n"
                
            if "high_volatility" in indicators.regime:
                analysis_log += "🌪️  HIGH VOLATILITY REGIME - Increased risk and opportunity\n"
                
            if abs(self.analysis_engine.get_cvd_divergence()) > 0.1:
                analysis_log += f"⚡ CVD DIVERGENCE - Spot/Perp flow showing {self.analysis_engine.get_cvd_divergence():.1%} divergence\n"
                
            analysis_log += "================================================================================\n"
            
            # Log the complete analysis
            logger.info(analysis_log)
            
            # Save to analysis file
            analysis_file = f"btc_analysis_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(analysis_file, 'w') as f:
                f.write(analysis_log)
                
            self.performance_metrics['total_trades_processed'] = self.trade_count
            
        except Exception as e:
            logger.error(f"Error generating detailed analysis: {e}")
            
    async def _shutdown(self):
        """Graceful shutdown of the analysis engine"""
        logger.info("Shutting down Market Analysis Engine...")
        
        try:
            # Stop data ingestion
            self.data_ingestor.stop_ingestion()
            
            # Generate final report
            if self.start_time:
                runtime = time.time() - self.start_time
                logger.info(f"Final Statistics:")
                logger.info(f"  - Runtime: {runtime/3600:.2f} hours")
                logger.info(f"  - Trades Processed: {self.trade_count:,}")
                logger.info(f"  - Analysis Updates: {self.performance_metrics['analysis_updates']:,}")
                logger.info(f"  - Alerts Generated: {self.performance_metrics['total_alerts_generated']:,}")
                logger.info(f"  - Errors: {self.performance_metrics['errors']:,}")
                
            logger.info("Market Analysis Engine shutdown complete.")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


class CLIDashboard:
    """
    Simple CLI dashboard for monitoring the analysis engine
    """
    
    def __init__(self, engine: MarketAnalysisEngine):
        self.engine = engine
        
    async def display_dashboard(self):
        """Display real-time dashboard in terminal"""
        while self.engine.running:
            try:
                # Clear screen (works on most terminals)
                os.system('clear' if os.name == 'posix' else 'cls')
                
                # Dashboard header
                print("=" * 80)
                print("🚀 REAL-TIME MARKET ANALYSIS DASHBOARD")
                print("=" * 80)
                
                if self.engine.current_indicators:
                    indicators = self.engine.current_indicators
                    
                    print(f"Symbol: {indicators.symbol}")
                    print(f"Price: ${self.engine.current_price:,.2f}")
                    print(f"CVD: {indicators.cvd:,.2f}")
                    print(f"Volatility: {indicators.volatility:.4f}")
                    print(f"Regime: {indicators.regime}")
                    print(f"Hurst: {indicators.hurst_exponent:.3f}")
                    print(f"Ichimoku: {indicators.ichimoku_signal}")
                    
                    # Recent alerts
                    recent_alerts = self.engine.pattern_detector.get_recent_alerts(10)
                    print(f"\nRecent Alerts ({len(recent_alerts)}):")
                    for alert in recent_alerts[-5:]:  # Show last 5
                        timestamp = datetime.fromtimestamp(alert.timestamp/1000).strftime('%H:%M:%S')
                        print(f"  {timestamp} - {alert.alert_type}: {alert.message[:50]}...")
                        
                print(f"\nPress Ctrl+C to stop")
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Dashboard error: {e}")
                await asyncio.sleep(5)


async def main():
    """Main application entry point"""
    logger.info("Initializing Real-Time Market Analysis Engine")
    
    # Configuration
    SYMBOL = "BTCUSDT"
    LOG_INTERVAL = 60  # seconds between detailed logs
    
    try:
        # Create analysis engine
        engine = MarketAnalysisEngine(symbol=SYMBOL, log_interval=LOG_INTERVAL)
        
        # Optional: Create CLI dashboard
        dashboard = CLIDashboard(engine)
        
        # Start both engine and dashboard
        tasks = [
            asyncio.create_task(engine.start_analysis()),
            # asyncio.create_task(dashboard.display_dashboard())  # Uncomment for CLI dashboard
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Critical application error: {e}")
    finally:
        logger.info("Application shutdown complete")


if __name__ == "__main__":
    # Run the main application
    asyncio.run(main())
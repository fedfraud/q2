# Real-Time Market Analysis Engine

A comprehensive Python-based real-time cryptocurrency market analysis system that connects to live exchange data streams and performs advanced mathematical and statistical analysis for trading insights.

## Overview

This engine provides real-time analysis of cryptocurrency markets with the following capabilities:

- **Real-time data ingestion** from Binance WebSocket APIs
- **Advanced mathematical analysis** including CVD, Kalman filtering, GARCH modeling, and more
- **Pattern detection** for bull/bear traps, liquidation cascades, and regime shifts
- **Professional-grade alerting** system with confidence scoring
- **Comprehensive logging** and analysis reporting

## Core Components

### 1. Data Ingestion Module (`data_ingestor.py`)
- WebSocket connections to Binance for real-time trade and order book data
- Real-time CVD (Cumulative Volume Delta) calculation
- Order flow analysis and volume tracking
- Market depth analysis from order book data

### 2. Analysis Engine Module (`analysis_engine.py`)
- **Kalman Filter** for signal smoothing
- **GARCH(1,1) model** for conditional volatility estimation
- **Hidden Markov Model-like** regime detection
- **Hurst Exponent** calculation for trend persistence analysis
- **Ichimoku Cloud** indicators for trend analysis
- Real-time mathematical indicator updates

### 3. Pattern Detection Module (`pattern_detector.py`)
- **Bull/Bear Trap Detection** using CVD divergence analysis
- **Liquidation Cascade Alerts** based on volume and volatility patterns
- **Market Regime Shift Detection** for volatility transitions
- **Volume Anomaly Detection** using statistical analysis
- Confidence-based alert scoring system

### 4. Main Application (`main.py`)
- Orchestrates all modules in real-time
- Professional analysis logging and reporting
- High-priority alert management
- Performance metrics tracking
- Graceful shutdown handling

## Features

- ✅ Real-time WebSocket data streaming
- ✅ Mathematical analysis with multiple models
- ✅ Pattern recognition and alerting
- ✅ Professional logging and reporting
- ✅ Modular, extensible architecture
- ✅ Error handling and recovery
- ✅ Performance monitoring

## Installation

1. Clone the repository:
```bash
git clone https://github.com/fedfraud/q2.git
cd q2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Note: If you encounter dependency issues, the core system works with standard Python libraries. The advanced features require additional packages.

## Usage

### Running the Complete System

Start the real-time analysis engine:
```bash
python main.py
```

This will:
- Connect to Binance WebSocket streams for BTC/USDT
- Begin real-time mathematical analysis
- Generate alerts when patterns are detected
- Log detailed analysis every 60 seconds
- Save high-priority alerts to JSON files

### Testing the System

Run the test suite to verify functionality:
```bash
python test_system.py
```

This simulates market data and tests all components without requiring live connections.

### Configuration

Key parameters can be modified in `main.py`:
- `SYMBOL`: Trading pair to analyze (default: "BTCUSDT")
- `LOG_INTERVAL`: Seconds between detailed analysis logs (default: 60)

## Sample Output

### Real-time Analysis Log
```
================================================================================
REAL-TIME MARKET ANALYSIS REPORT
================================================================================
Symbol: BTCUSDT
Timestamp: 2025-01-14 15:30:45 UTC
Current Price: $50,247.82

MATHEMATICAL INDICATORS:
├─ Cumulative Volume Delta (CVD): 1,234.56
├─ CVD (Kalman Smoothed): 1,198.34
├─ Conditional Volatility (GARCH): 0.0456
├─ Market Regime: high_volatility_trending
├─ Hurst Exponent: 0.723
├─ Ichimoku Signal: bullish
└─ Trend Strength: 0.812

PATTERN ANALYSIS:
├─ CVD Divergence: -0.156
├─ Recent Alerts (30min): 3
└─ Alert Summary: {'BULL_TRAP': 1, 'REGIME_SHIFT': 1, 'VOLUME_ANOMALY_HIGH': 1}

MARKET INTERPRETATION:
⚠️  ELEVATED VOLATILITY - Market showing increased uncertainty
📈 TRENDING MARKET - Strong directional persistence detected
🌪️  HIGH VOLATILITY REGIME - Increased risk and opportunity
⚡ CVD DIVERGENCE - Spot/Perp flow showing -15.6% divergence
================================================================================
```

### Alert Examples
```
🚨 HIGH ALERT: BULL_TRAP - Potential Bull Trap Detected - Price rising 2.34% but CVD diverging -18.7%
   Confidence: 0.87, Symbol: BTCUSDT

🚨 CRITICAL ALERT: LIQUIDATION_CASCADE - Liquidation Cascade Alert - $250M approaching at $50000
   Confidence: 0.93, Symbol: BTCUSDT
```

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Data Ingestor  │───▶│ Analysis Engine │───▶│Pattern Detector │
│                 │    │                 │    │                 │
│ • WebSocket     │    │ • Kalman Filter │    │ • Bull/Bear     │
│ • CVD Calc      │    │ • GARCH Model   │    │ • Liquidations  │
│ • Order Flow    │    │ • HMM Regime    │    │ • Regime Shifts │
│ • Market Depth  │    │ • Hurst Exp     │    │ • Volume Anom   │
└─────────────────┘    │ • Ichimoku      │    └─────────────────┘
                       └─────────────────┘             │
                                │                      │
                                ▼                      ▼
                       ┌─────────────────────────────────────┐
                       │           Main Engine               │
                       │                                     │
                       │ • Orchestration  • Professional    │
                       │ • Logging        • Alert System    │
                       │ • Metrics        • File Output     │
                       └─────────────────────────────────────┘
```

## Mathematical Models

### 1. Kalman Filter
Smooths CVD signals to reduce noise and improve trend detection.

### 2. GARCH(1,1) Model
Estimates conditional volatility using the formula:
```
σ²(t) = ω + α·ε²(t-1) + β·σ²(t-1)
```

### 3. Hurst Exponent
Measures trend persistence:
- H > 0.5: Trending behavior
- H < 0.5: Mean-reverting behavior
- H = 0.5: Random walk

### 4. Market Regime Detection
Classifies market states:
- `low_volatility_ranging`
- `low_volatility_trending`
- `high_volatility_choppy`
- `high_volatility_trending`

## File Structure

```
q2/
├── data_ingestor.py      # WebSocket data ingestion
├── analysis_engine.py    # Mathematical analysis models
├── pattern_detector.py   # Pattern recognition and alerts
├── main.py              # Main application orchestrator
├── test_system.py       # Test suite
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── [Generated Files]
    ├── market_analysis.log           # Runtime logs
    ├── high_priority_alerts_*.json   # Critical alerts
    └── btc_analysis_complete_*.txt   # Detailed analysis reports
```

## Alert Types

- **BULL_TRAP**: Price rising with negative CVD divergence
- **BEAR_TRAP**: Price falling with positive CVD divergence
- **LIQUIDATION_CASCADE**: High liquidation risk detected
- **REGIME_SHIFT**: Market volatility regime change
- **VOLUME_ANOMALY**: Unusual volume patterns detected

## Extending the System

### Adding New Exchanges
1. Create a new WebSocket client in `data_ingestor.py`
2. Implement exchange-specific data parsing
3. Update the aggregator to handle multiple sources

### Adding New Indicators
1. Add the mathematical model to `analysis_engine.py`
2. Update the `MarketIndicators` dataclass
3. Modify the analysis update loop

### Adding New Patterns
1. Create a new detector class in `pattern_detector.py`
2. Add the detector to the `PatternDetector` main class
3. Define new alert types and messages

## Performance Considerations

- The system processes real-time data with minimal latency
- Memory usage is controlled with fixed-size deques
- Error handling prevents system crashes
- Configurable logging intervals balance detail with performance

## Production Deployment

For production use, consider:
- Database integration for persistent storage
- Monitoring and health checks
- Load balancing for multiple symbols
- API endpoints for external system integration
- Enhanced security for WebSocket connections

## Disclaimer

This software is for educational and research purposes. Cryptocurrency trading involves significant financial risk. Always conduct your own analysis and risk management before making trading decisions.

## License

This project is open source. See the repository for license details.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for improvements.
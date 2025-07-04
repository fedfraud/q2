"""
Data Ingestion Module for Real-Time Market Analysis Engine

This module handles real-time data ingestion from cryptocurrency exchanges,
specifically focusing on Binance WebSocket connections for:
- Trade data (for CVD calculation)
- Order book data (for market depth analysis)
- Liquidation data (for liquidation cascade detection)
- Funding rate data (for leverage cost tracking)
"""

import asyncio
import websockets
import json
import logging
from datetime import datetime
from typing import Dict, List, Callable, Optional
import time
from collections import defaultdict
import threading
import queue

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BinanceWebSocketClient:
    """
    WebSocket client for Binance exchange data streams
    """
    
    def __init__(self):
        self.base_url = "wss://stream.binance.com:9443/ws/"
        self.connections = {}
        self.callbacks = defaultdict(list)
        self.running = False
        
    def add_callback(self, stream: str, callback: Callable):
        """Add callback function for a specific data stream"""
        self.callbacks[stream].append(callback)
        
    async def connect_stream(self, stream: str):
        """Connect to a specific WebSocket stream"""
        try:
            url = f"{self.base_url}{stream}"
            logger.info(f"Connecting to {url}")
            
            async with websockets.connect(url) as websocket:
                self.connections[stream] = websocket
                logger.info(f"Connected to {stream}")
                
                async for message in websocket:
                    if not self.running:
                        break
                        
                    try:
                        data = json.loads(message)
                        # Call all registered callbacks for this stream
                        for callback in self.callbacks[stream]:
                            callback(data)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to decode JSON: {e}")
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                        
        except Exception as e:
            logger.error(f"Error connecting to {stream}: {e}")
            
    async def start_streams(self, streams: List[str]):
        """Start multiple WebSocket streams concurrently"""
        self.running = True
        tasks = []
        
        for stream in streams:
            task = asyncio.create_task(self.connect_stream(stream))
            tasks.append(task)
            
        await asyncio.gather(*tasks, return_exceptions=True)
        
    def stop(self):
        """Stop all WebSocket connections"""
        self.running = False
        logger.info("Stopping WebSocket connections")


class DataAggregator:
    """
    Aggregates and processes raw WebSocket data for analysis
    """
    
    def __init__(self):
        self.trade_data = queue.Queue()
        self.orderbook_data = queue.Queue()
        self.liquidation_data = queue.Queue()
        self.funding_rate_data = queue.Queue()
        
        # CVD calculation variables
        self.cvd_cumulative = 0.0
        self.last_cvd_update = time.time()
        
        # Order flow tracking
        self.buy_volume = 0.0
        self.sell_volume = 0.0
        
    def process_trade_data(self, data: Dict):
        """Process individual trade data for CVD calculation"""
        try:
            if 'data' in data:
                trade = data['data']
                
                # Extract trade information
                symbol = trade.get('s', '')
                price = float(trade.get('p', 0))
                quantity = float(trade.get('q', 0))
                is_buyer_maker = trade.get('m', False)  # True if buyer is maker (sell order)
                timestamp = trade.get('T', 0)
                
                # Calculate volume delta
                volume_delta = quantity if not is_buyer_maker else -quantity
                self.cvd_cumulative += volume_delta
                
                # Track buy/sell volumes
                if not is_buyer_maker:  # Buy order
                    self.buy_volume += quantity
                else:  # Sell order
                    self.sell_volume += quantity
                
                # Store processed trade data
                processed_trade = {
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'price': price,
                    'quantity': quantity,
                    'side': 'buy' if not is_buyer_maker else 'sell',
                    'cvd': self.cvd_cumulative,
                    'volume_delta': volume_delta
                }
                
                self.trade_data.put(processed_trade)
                
                # Log significant CVD changes
                current_time = time.time()
                if current_time - self.last_cvd_update > 5:  # Log every 5 seconds
                    logger.info(f"CVD Update - Symbol: {symbol}, CVD: {self.cvd_cumulative:.2f}, "
                              f"Buy Vol: {self.buy_volume:.2f}, Sell Vol: {self.sell_volume:.2f}")
                    self.last_cvd_update = current_time
                    
        except Exception as e:
            logger.error(f"Error processing trade data: {e}")
            
    def process_orderbook_data(self, data: Dict):
        """Process order book data for market depth analysis"""
        try:
            if 'data' in data:
                orderbook = data['data']
                
                # Extract bid/ask levels
                bids = orderbook.get('b', [])
                asks = orderbook.get('a', [])
                
                # Calculate market depth metrics
                bid_depth = sum(float(bid[1]) for bid in bids[:10])  # Top 10 levels
                ask_depth = sum(float(ask[1]) for ask in asks[:10])
                
                processed_orderbook = {
                    'timestamp': time.time() * 1000,
                    'symbol': orderbook.get('s', ''),
                    'bid_depth': bid_depth,
                    'ask_depth': ask_depth,
                    'depth_ratio': bid_depth / ask_depth if ask_depth > 0 else 0,
                    'spread': float(asks[0][0]) - float(bids[0][0]) if bids and asks else 0
                }
                
                self.orderbook_data.put(processed_orderbook)
                
        except Exception as e:
            logger.error(f"Error processing orderbook data: {e}")
            
    def get_latest_data(self, data_type: str, max_items: int = 100) -> List[Dict]:
        """Get latest data of specified type"""
        data_queue = getattr(self, f"{data_type}_data", None)
        if not data_queue:
            return []
            
        items = []
        try:
            while len(items) < max_items and not data_queue.empty():
                items.append(data_queue.get_nowait())
        except queue.Empty:
            pass
            
        return items
        
    def get_current_cvd(self) -> float:
        """Get current cumulative volume delta"""
        return self.cvd_cumulative
        
    def reset_cvd(self):
        """Reset CVD calculation (useful for daily resets)"""
        self.cvd_cumulative = 0.0
        self.buy_volume = 0.0
        self.sell_volume = 0.0
        logger.info("CVD reset")


class MarketDataIngestor:
    """
    Main class for coordinating market data ingestion
    """
    
    def __init__(self, symbol: str = "btcusdt"):
        self.symbol = symbol.lower()
        self.ws_client = BinanceWebSocketClient()
        self.aggregator = DataAggregator()
        
        # Setup stream callbacks
        self.setup_callbacks()
        
    def setup_callbacks(self):
        """Setup callbacks for different data streams"""
        # Trade stream for CVD calculation
        trade_stream = f"{self.symbol}@trade"
        self.ws_client.add_callback(trade_stream, self.aggregator.process_trade_data)
        
        # Order book stream for market depth
        depth_stream = f"{self.symbol}@depth20@100ms"
        self.ws_client.add_callback(depth_stream, self.aggregator.process_orderbook_data)
        
    async def start_ingestion(self):
        """Start real-time data ingestion"""
        streams = [
            f"{self.symbol}@trade",
            f"{self.symbol}@depth20@100ms"
        ]
        
        logger.info(f"Starting market data ingestion for {self.symbol.upper()}")
        await self.ws_client.start_streams(streams)
        
    def stop_ingestion(self):
        """Stop data ingestion"""
        self.ws_client.stop()
        
    def get_aggregator(self) -> DataAggregator:
        """Get the data aggregator for analysis"""
        return self.aggregator


# Example usage and testing
if __name__ == "__main__":
    async def main():
        # Create data ingestor for BTC/USDT
        ingestor = MarketDataIngestor("BTCUSDT")
        
        # Start data ingestion
        try:
            await ingestor.start_ingestion()
        except KeyboardInterrupt:
            logger.info("Shutting down data ingestion")
            ingestor.stop_ingestion()
    
    # Run the data ingestor
    asyncio.run(main())
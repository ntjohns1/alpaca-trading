from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from dotenv import load_dotenv
import os

load_dotenv()

class AlpacaClient:
    def __init__(self):
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        is_paper = os.getenv('ALPACA_PAPER', 'True').lower() == 'true'
        
        self.trading_client = TradingClient(api_key, secret_key, paper=is_paper)
        self.data_client = StockHistoricalDataClient(api_key, secret_key)

    def get_account(self):
        """Get account information"""
        return self.trading_client.get_account()

    def place_order(self, symbol, qty, side):
        """Place a market order"""
        order_data = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY
        )
        return self.trading_client.submit_order(order_data)

    def get_historical_data(self, symbol, timeframe, start_date, end_date):
        """Get historical market data"""
        return self.data_client.get_stock_bars(
            symbol=symbol,
            timeframe=timeframe,
            start=start_date,
            end=end_date
        )

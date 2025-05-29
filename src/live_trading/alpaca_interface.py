"""
Alpaca Interface for RL Trading Agent

This module provides an interface for RL agents to interact with Alpaca API
for live trading with appropriate safety mechanisms.
"""

import os
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# Configure logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class AlpacaInterface:
    """
    Interface between RL agent and Alpaca API for live trading
    
    Features:
    - Safety limits for maximum position sizes and drawdowns
    - Circuit breakers to pause trading during extreme market conditions
    - Monitoring of agent performance
    - Conversion between agent actions and actual orders
    - Historical data retrieval for agent state
    """
    
    def __init__(self, 
                 max_position_value=10000,
                 max_single_order_value=1000,
                 max_daily_drawdown_pct=5.0,
                 max_total_drawdown_pct=10.0,
                 paper_trading=True):
        """
        Initialize the Alpaca interface
        
        Parameters:
        -----------
        max_position_value : float
            Maximum value of a single position in dollars
        max_single_order_value : float
            Maximum value of a single order in dollars
        max_daily_drawdown_pct : float
            Maximum allowed daily drawdown as percentage
        max_total_drawdown_pct : float
            Maximum allowed total drawdown as percentage
        paper_trading : bool
            Whether to use paper trading
        """
        # Load environment variables
        load_dotenv()
        
        # Safety parameters
        self.max_position_value = max_position_value
        self.max_single_order_value = max_single_order_value
        self.max_daily_drawdown_pct = max_daily_drawdown_pct / 100
        self.max_total_drawdown_pct = max_total_drawdown_pct / 100
        
        # API credentials
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.api_secret = os.getenv('ALPACA_API_SECRET')
        self.api_url = os.getenv('ALPACA_API_URL')
        
        if not self.api_key or not self.api_secret:
            raise ValueError("Alpaca API credentials not found in environment variables")
        
        # Initialize clients
        self.trading_client = TradingClient(self.api_key, self.api_secret, paper=paper_trading)
        self.data_client = StockHistoricalDataClient(self.api_key, self.api_secret)
        
        # Performance tracking
        self.initial_equity = None
        self.previous_equity = None
        self.daily_high_equity = None
        self.start_date = datetime.now()
        
        # Trading state
        self.is_trading_allowed = True
        self.circuit_breaker_triggered = False
        self.positions = {}
        
        # Initialize account
        self._initialize_account()
    
    def _initialize_account(self):
        """Initialize account information and set baseline equity"""
        account = self.trading_client.get_account()
        self.initial_equity = float(account.equity)
        self.previous_equity = self.initial_equity
        self.daily_high_equity = self.initial_equity
        
        log.info(f"Account initialized with equity: ${self.initial_equity:.2f}")
        log.info(f"Trading limits: Max position ${self.max_position_value}, " +
                 f"Max order ${self.max_single_order_value}")
        log.info(f"Circuit breakers: Daily drawdown {self.max_daily_drawdown_pct:.1%}, " +
                 f"Total drawdown {self.max_total_drawdown_pct:.1%}")
    
    def check_circuit_breakers(self):
        """
        Check if any circuit breakers should be triggered
        
        Returns:
        --------
        bool : Whether trading is allowed
        """
        # Skip if already triggered
        if self.circuit_breaker_triggered:
            return False
        
        # Get current account equity
        account = self.trading_client.get_account()
        current_equity = float(account.equity)
        
        # Update daily high water mark
        self.daily_high_equity = max(self.daily_high_equity, current_equity)
        
        # Calculate drawdowns
        daily_drawdown = (self.daily_high_equity - current_equity) / self.daily_high_equity
        total_drawdown = (self.initial_equity - current_equity) / self.initial_equity
        
        # Check circuit breakers
        if daily_drawdown > self.max_daily_drawdown_pct:
            log.warning(f"Daily drawdown circuit breaker triggered: {daily_drawdown:.2%}")
            self.circuit_breaker_triggered = True
            self.is_trading_allowed = False
            return False
        
        if total_drawdown > self.max_total_drawdown_pct:
            log.warning(f"Total drawdown circuit breaker triggered: {total_drawdown:.2%}")
            self.circuit_breaker_triggered = True
            self.is_trading_allowed = False
            return False
        
        # Update previous equity
        self.previous_equity = current_equity
        return True
    
    def reset_daily_circuit_breakers(self):
        """Reset daily circuit breakers (call at market open)"""
        account = self.trading_client.get_account()
        current_equity = float(account.equity)
        
        self.daily_high_equity = current_equity
        
        # Only reset circuit breaker if it was triggered due to daily drawdown
        if self.circuit_breaker_triggered:
            total_drawdown = (self.initial_equity - current_equity) / self.initial_equity
            if total_drawdown < self.max_total_drawdown_pct:
                self.circuit_breaker_triggered = False
                self.is_trading_allowed = True
                log.info("Daily circuit breakers reset")
    
    def get_account_info(self):
        """
        Get current account information
        
        Returns:
        --------
        dict : Account information
        """
        account = self.trading_client.get_account()
        
        # Calculate performance metrics
        current_equity = float(account.equity)
        daily_return = (current_equity - self.previous_equity) / self.previous_equity
        total_return = (current_equity - self.initial_equity) / self.initial_equity
        
        return {
            'equity': current_equity,
            'cash': float(account.cash),
            'buying_power': float(account.buying_power),
            'daily_return': daily_return,
            'total_return': total_return,
            'is_trading_allowed': self.is_trading_allowed,
            'circuit_breaker_triggered': self.circuit_breaker_triggered
        }
    
    def get_positions(self):
        """
        Get current positions
        
        Returns:
        --------
        dict : Dictionary of positions by symbol
        """
        positions = self.trading_client.get_all_positions()
        
        result = {}
        for position in positions:
            result[position.symbol] = {
                'qty': float(position.qty),
                'market_value': float(position.market_value),
                'avg_entry_price': float(position.avg_entry_price),
                'current_price': float(position.current_price),
                'unrealized_pl': float(position.unrealized_pl),
                'unrealized_plpc': float(position.unrealized_plpc)
            }
        
        return result
    
    def get_historical_data(self, symbol, timeframe=TimeFrame.Day, limit=20):
        """
        Get historical market data for a symbol
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
        timeframe : TimeFrame
            Time frame for the data
        limit : int
            Number of bars to retrieve
            
        Returns:
        --------
        pd.DataFrame : Historical data
        """
        # Calculate start and end dates
        end = datetime.now()
        
        # Request parameters
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe,
            limit=limit
        )
        
        # Get the data
        bars = self.data_client.get_stock_bars(request_params)
        
        # Convert to DataFrame if we got data
        if bars and symbol in bars:
            df = bars[symbol].df
            return df
        
        return pd.DataFrame()
    
    def get_state_features(self, symbols, lookback=20):
        """
        Get state features for the RL agent
        
        Parameters:
        -----------
        symbols : list
            List of stock symbols
        lookback : int
            Number of days of history to include
            
        Returns:
        --------
        dict : State features by symbol
        """
        features = {}
        
        for symbol in symbols:
            # Get historical data
            df = self.get_historical_data(symbol, limit=lookback)
            
            if df.empty:
                log.warning(f"No data available for {symbol}")
                continue
            
            # Calculate features (similar to trading_env.py)
            df['returns'] = df['close'].pct_change()
            df['ret_2'] = df['close'].pct_change(2)
            df['ret_5'] = df['close'].pct_change(5)
            df['ret_10'] = df['close'].pct_change(10)
            
            # Add more features as needed
            # For example, moving averages, RSI, etc.
            
            # Store the features
            features[symbol] = df.dropna().iloc[-1].to_dict()
            
            # Add position information if available
            positions = self.get_positions()
            if symbol in positions:
                features[symbol]['position'] = positions[symbol]['qty']
                features[symbol]['position_value'] = positions[symbol]['market_value']
                features[symbol]['unrealized_pl'] = positions[symbol]['unrealized_pl']
            else:
                features[symbol]['position'] = 0
                features[symbol]['position_value'] = 0
                features[symbol]['unrealized_pl'] = 0
        
        return features
    
    def execute_action(self, symbol, action, quantity=None, value=None, limit_price=None):
        """
        Execute a trading action
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
        action : int or str
            0/SHORT: Short position
            1/HOLD: Hold/do nothing
            2/LONG: Long position
        quantity : float
            Number of shares (optional)
        value : float
            Dollar value of the order (optional)
        limit_price : float
            Limit price for the order (optional)
            
        Returns:
        --------
        dict : Order information
        """
        # Check if trading is allowed
        if not self.is_trading_allowed:
            log.warning(f"Trading is not allowed - circuit breaker triggered")
            return {'success': False, 'message': 'Circuit breaker triggered'}
        
        # Check circuit breakers
        if not self.check_circuit_breakers():
            return {'success': False, 'message': 'Circuit breaker check failed'}
        
        # Convert action to order side
        if isinstance(action, int):
            if action == 0:  # SHORT
                side = OrderSide.SELL
            elif action == 1:  # HOLD
                return {'success': True, 'message': 'Hold position - no action taken'}
            elif action == 2:  # LONG
                side = OrderSide.BUY
            else:
                return {'success': False, 'message': f'Invalid action: {action}'}
        else:
            action = action.upper()
            if action in ['SHORT', 'SELL']:
                side = OrderSide.SELL
            elif action in ['HOLD', 'NONE']:
                return {'success': True, 'message': 'Hold position - no action taken'}
            elif action in ['LONG', 'BUY']:
                side = OrderSide.BUY
            else:
                return {'success': False, 'message': f'Invalid action: {action}'}
        
        # Get current positions
        positions = self.get_positions()
        current_position = positions.get(symbol, {'qty': 0})
        
        # Determine quantity and validate against limits
        if quantity is None and value is None:
            # Default to using max_single_order_value
            value = self.max_single_order_value
        
        if value is not None:
            # Get current price to calculate quantity
            latest_data = self.get_historical_data(symbol, limit=1)
            if latest_data.empty:
                return {'success': False, 'message': f'Could not get price for {symbol}'}
            
            current_price = latest_data['close'].iloc[-1]
            quantity = value / current_price
        
        # Validate against position limits
        if side == OrderSide.BUY:
            # Check if this would exceed max position value
            new_position_value = (current_position['qty'] + quantity) * current_price
            if new_position_value > self.max_position_value:
                log.warning(f"Order would exceed max position value: ${new_position_value:.2f}")
                # Adjust quantity to respect limit
                quantity = (self.max_position_value / current_price) - current_position['qty']
                if quantity <= 0:
                    return {'success': False, 'message': 'Position limit reached'}
        
        # Create order
        try:
            if limit_price:
                # Limit order
                order_data = LimitOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=limit_price
                )
            else:
                # Market order
                order_data = MarketOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=side,
                    time_in_force=TimeInForce.DAY
                )
            
            # Submit order
            order = self.trading_client.submit_order(order_data)
            
            log.info(f"Order submitted: {side.name} {quantity} shares of {symbol}")
            
            return {
                'success': True,
                'order_id': order.id,
                'symbol': symbol,
                'side': side.name,
                'qty': quantity,
                'type': 'LIMIT' if limit_price else 'MARKET'
            }
            
        except Exception as e:
            log.error(f"Error submitting order: {e}")
            return {'success': False, 'message': str(e)}
    
    def cancel_order(self, order_id):
        """
        Cancel an open order
        
        Parameters:
        -----------
        order_id : str
            Order ID to cancel
            
        Returns:
        --------
        bool : Whether the cancellation was successful
        """
        try:
            self.trading_client.cancel_order(order_id)
            log.info(f"Order {order_id} canceled")
            return True
        except Exception as e:
            log.error(f"Error canceling order {order_id}: {e}")
            return False
    
    def cancel_all_orders(self):
        """
        Cancel all open orders
        
        Returns:
        --------
        int : Number of orders canceled
        """
        try:
            orders = self.trading_client.get_orders()
            count = 0
            
            for order in orders:
                if order.status == OrderStatus.NEW or order.status == OrderStatus.PARTIALLY_FILLED:
                    self.trading_client.cancel_order(order.id)
                    count += 1
            
            log.info(f"Canceled {count} orders")
            return count
        except Exception as e:
            log.error(f"Error canceling orders: {e}")
            return 0
    
    def liquidate_position(self, symbol):
        """
        Liquidate a position
        
        Parameters:
        -----------
        symbol : str
            Symbol to liquidate
            
        Returns:
        --------
        dict : Order information
        """
        try:
            positions = self.get_positions()
            if symbol not in positions:
                return {'success': False, 'message': f'No position for {symbol}'}
            
            position = positions[symbol]
            qty = abs(float(position['qty']))
            
            if qty == 0:
                return {'success': False, 'message': f'No position for {symbol}'}
            
            # Determine side (opposite of current position)
            side = OrderSide.SELL if float(position['qty']) > 0 else OrderSide.BUY
            
            # Create market order
            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY
            )
            
            # Submit order
            order = self.trading_client.submit_order(order_data)
            
            log.info(f"Liquidated position: {side.name} {qty} shares of {symbol}")
            
            return {
                'success': True,
                'order_id': order.id,
                'symbol': symbol,
                'side': side.name,
                'qty': qty,
                'type': 'MARKET'
            }
            
        except Exception as e:
            log.error(f"Error liquidating position: {e}")
            return {'success': False, 'message': str(e)}
    
    def liquidate_all_positions(self):
        """
        Liquidate all positions
        
        Returns:
        --------
        dict : Results by symbol
        """
        try:
            positions = self.get_positions()
            results = {}
            
            for symbol in positions:
                result = self.liquidate_position(symbol)
                results[symbol] = result
            
            log.info(f"Liquidated {len(results)} positions")
            return results
        except Exception as e:
            log.error(f"Error liquidating positions: {e}")
            return {'success': False, 'message': str(e)}

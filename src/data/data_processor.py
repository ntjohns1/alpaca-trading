"""
Data Processing Module for Alpaca Trading

This module handles the ETL (Extract, Transform, Load) pipeline for market data:
1. Extract: Download historical data from Alpaca API
2. Transform: Clean and process the data, calculate features
3. Load: Store the processed data in HDF5 files for efficient access
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import h5py
import tables
import talib
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv

# Configure logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class DataProcessor:
    """
    Data processor for downloading, cleaning, and storing market data
    """
    
    def __init__(self, data_dir=None):
        """
        Initialize the data processor
        
        Parameters:
        -----------
        data_dir : str or Path
            Directory to store data files
        """
        # Load environment variables
        load_dotenv()
        
        # API credentials
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.api_secret = os.getenv('ALPACA_API_SECRET')
        
        if not self.api_key or not self.api_secret:
            raise ValueError("Alpaca API credentials not found in environment variables")
        
        # Initialize client
        self.data_client = StockHistoricalDataClient(self.api_key, self.api_secret)
        
        # Data directory
        self.data_dir = Path(data_dir) if data_dir else Path('data')
        self.data_dir.mkdir(exist_ok=True)
        
    def download_historical_data(self, symbols, timeframe=TimeFrame.Day, 
                                start_date=None, end_date=None, limit=None):
        """
        Download historical data from Alpaca API
        
        Parameters:
        -----------
        symbols : list
            List of stock symbols
        timeframe : TimeFrame
            Time frame for the data
        start_date : datetime
            Start date for the data
        end_date : datetime
            End date for the data
        limit : int
            Number of bars to retrieve (used if start_date is None)
            
        Returns:
        --------
        dict : Dictionary of DataFrames by symbol
        """
        if not symbols:
            raise ValueError("No symbols provided")
        
        # Set default end date to today
        if end_date is None:
            end_date = datetime.now()
        
        # Set default start date if not provided
        if start_date is None and limit is not None:
            if timeframe == TimeFrame.Day:
                start_date = end_date - timedelta(days=limit * 1.5)  # Add buffer for weekends/holidays
            elif timeframe == TimeFrame.Hour:
                start_date = end_date - timedelta(hours=limit * 1.5)
            elif timeframe == TimeFrame.Minute:
                start_date = end_date - timedelta(minutes=limit * 1.5)
        
        # Request parameters
        request_params = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=timeframe,
            start=start_date,
            end=end_date
        )
        
        log.info(f"Downloading data for {len(symbols)} symbols from {start_date} to {end_date}")
        
        try:
            # Get the data
            bars = self.data_client.get_stock_bars(request_params)
            
            # Convert to dictionary of DataFrames
            data = {}
            for symbol in symbols:
                if symbol in bars:
                    data[symbol] = bars[symbol].df
                    log.info(f"Downloaded {len(data[symbol])} bars for {symbol}")
                else:
                    log.warning(f"No data available for {symbol}")
            
            return data
            
        except Exception as e:
            log.error(f"Error downloading data: {e}")
            return {}
    
    def calculate_features(self, data):
        """
        Calculate technical indicators and features
        
        Parameters:
        -----------
        data : DataFrame
            Price data with OHLCV columns
            
        Returns:
        --------
        DataFrame : Data with additional features
        """
        if data.empty:
            return data
        
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Basic returns
        df['returns'] = df['close'].pct_change()
        df['ret_2'] = df['close'].pct_change(2)
        df['ret_5'] = df['close'].pct_change(5)
        df['ret_10'] = df['close'].pct_change(10)
        df['ret_21'] = df['close'].pct_change(21)
        
        # Volume features
        if 'volume' in df.columns:
            df['volume_ma5'] = df['volume'].rolling(5).mean()
            df['volume_ma10'] = df['volume'].rolling(10).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma5']
        
        # Price-based indicators
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # RSI
            df['rsi'] = talib.RSI(df['close'])
            
            # MACD
            macd, signal, hist = talib.MACD(df['close'])
            df['macd'] = macd
            df['macd_signal'] = signal
            df['macd_hist'] = hist
            
            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(df['close'])
            df['bb_upper'] = upper
            df['bb_middle'] = middle
            df['bb_lower'] = lower
            df['bb_width'] = (upper - lower) / middle
            
            # ATR
            df['atr'] = talib.ATR(df['high'], df['low'], df['close'])
            
            # Stochastic
            slowk, slowd = talib.STOCH(df['high'], df['low'], df['close'])
            df['stoch_k'] = slowk
            df['stoch_d'] = slowd
            
            # Moving Averages
            df['sma5'] = talib.SMA(df['close'], timeperiod=5)
            df['sma10'] = talib.SMA(df['close'], timeperiod=10)
            df['sma20'] = talib.SMA(df['close'], timeperiod=20)
            df['sma50'] = talib.SMA(df['close'], timeperiod=50)
            
            # Moving Average Crossovers
            df['sma5_10_cross'] = df['sma5'] - df['sma10']
            df['sma10_20_cross'] = df['sma10'] - df['sma20']
            
            # Momentum
            df['mom10'] = talib.MOM(df['close'], timeperiod=10)
            
            # ADX - Trend strength
            df['adx'] = talib.ADX(df['high'], df['low'], df['close'])
        
        # Drop NaN values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        return df
    
    def process_data(self, data):
        """
        Process downloaded data: calculate features and clean
        
        Parameters:
        -----------
        data : dict
            Dictionary of DataFrames by symbol
            
        Returns:
        --------
        dict : Dictionary of processed DataFrames
        """
        processed_data = {}
        
        for symbol, df in data.items():
            if df.empty:
                continue
                
            # Calculate features
            processed_df = self.calculate_features(df)
            
            # Drop NaN values
            processed_df = processed_df.dropna()
            
            if processed_df.empty:
                log.warning(f"No valid data after processing for {symbol}")
                continue
                
            processed_data[symbol] = processed_df
            log.info(f"Processed {len(processed_df)} rows for {symbol}")
        
        return processed_data
    
    def save_to_hdf5(self, data, filename='market_data.h5'):
        """
        Save processed data to HDF5 file
        
        Parameters:
        -----------
        data : dict
            Dictionary of DataFrames by symbol
        filename : str
            Name of the HDF5 file
            
        Returns:
        --------
        str : Path to the saved file
        """
        file_path = self.data_dir / filename
        
        try:
            with pd.HDFStore(file_path, mode='w') as store:
                for symbol, df in data.items():
                    if df.empty:
                        continue
                        
                    # Save to HDF5 file
                    store[f'data/{symbol}'] = df
                    log.info(f"Saved {len(df)} rows for {symbol} to {file_path}")
            
            return str(file_path)
            
        except Exception as e:
            log.error(f"Error saving data to HDF5: {e}")
            return None
    
    def load_from_hdf5(self, filename='market_data.h5', symbols=None):
        """
        Load data from HDF5 file
        
        Parameters:
        -----------
        filename : str
            Name of the HDF5 file
        symbols : list
            List of symbols to load (None for all)
            
        Returns:
        --------
        dict : Dictionary of DataFrames by symbol
        """
        file_path = self.data_dir / filename
        
        if not file_path.exists():
            log.error(f"File not found: {file_path}")
            return {}
        
        try:
            data = {}
            with pd.HDFStore(file_path, mode='r') as store:
                # Get available symbols
                keys = store.keys()
                
                # Filter by requested symbols if provided
                if symbols:
                    available_symbols = [key.split('/')[-1] for key in keys if key.startswith('/data/')]
                    symbols_to_load = [s for s in symbols if s in available_symbols]
                else:
                    symbols_to_load = [key.split('/')[-1] for key in keys if key.startswith('/data/')]
                
                # Load data for each symbol
                for symbol in symbols_to_load:
                    data[symbol] = store[f'data/{symbol}']
                    log.info(f"Loaded {len(data[symbol])} rows for {symbol} from {file_path}")
            
            return data
            
        except Exception as e:
            log.error(f"Error loading data from HDF5: {e}")
            return {}
    
    def run_etl_pipeline(self, symbols, timeframe=TimeFrame.Day, 
                        start_date=None, end_date=None, limit=None,
                        filename='market_data.h5'):
        """
        Run the full ETL pipeline: download, process, and save data
        
        Parameters:
        -----------
        symbols : list
            List of stock symbols
        timeframe : TimeFrame
            Time frame for the data
        start_date : datetime
            Start date for the data
        end_date : datetime
            End date for the data
        limit : int
            Number of bars to retrieve (used if start_date is None)
        filename : str
            Name of the HDF5 file
            
        Returns:
        --------
        dict : Dictionary of processed DataFrames
        """
        # Extract: Download data
        raw_data = self.download_historical_data(
            symbols=symbols,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )
        
        if not raw_data:
            log.error("No data downloaded")
            return {}
        
        # Transform: Process data
        processed_data = self.process_data(raw_data)
        
        if not processed_data:
            log.error("No data processed")
            return {}
        
        # Load: Save to HDF5
        file_path = self.save_to_hdf5(processed_data, filename)
        
        if not file_path:
            log.error("Failed to save data")
        
        return processed_data

"""
The MIT License (MIT)

Copyright (c) 2016 Tito Ingargiola
Copyright (c) 2019 Stefan Jansen
Copyright (c) 2025 Nelson (Updated version)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import sys
import os
import logging

import gym
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
from gym.envs.registration import register
from sklearn.preprocessing import scale
import talib

# Add the project root to the Python path
project_root = '/home/noslen/alpaca-trading'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set up logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.info('%s logger started.', __name__)


class DataSource:
    """
    Data source for TradingEnvironment

    Loads & preprocesses daily price & volume data
    Provides data for each new episode.
    
    This version only uses real data from the HDF5 file and includes
    improved error handling and data loading strategies.
    """

    def __init__(self, trading_days=252, ticker='AAPL', normalize=True):
        self.ticker = ticker
        self.trading_days = trading_days
        self.normalize = normalize
        self.data = self.load_data()
        self.preprocess_data()
        self.min_values = self.data.min().values.astype(np.float32)
        self.max_values = self.data.max().values.astype(np.float32)
        self.step = 0
        self.offset = None

    def load_data(self):
        """Load data for a specific ticker from HDF5 file"""
        log.info('Loading data for %s...', self.ticker)
        
        # Path to the HDF5 file
        hdf_path = '/home/noslen/alpaca-trading/data/assets.h5'
        
        # Check if the file exists
        if not os.path.exists(hdf_path):
            raise FileNotFoundError(f"HDF5 file not found at {hdf_path}")
        
        with pd.HDFStore(hdf_path, mode='r') as store:
            keys = store.keys()
            log.info(f"Available keys in HDF5 store: {keys}")
            
            if '/sharadar/prices' in keys:
                key = '/sharadar/prices'
            else:
                raise ValueError(f"No suitable price data found in {hdf_path}")
            
            # Check if our ticker exists and get data
            sample = store.select(key, start=0, stop=5)
            log.info(f"Sample data columns: {sample.columns}")
            log.info(f"Sample data index names: {sample.index.names}")
            
            if not isinstance(sample.index, pd.MultiIndex) or 'ticker' not in sample.index.names:
                raise ValueError(f"Data structure not as expected. Index should be MultiIndex with 'ticker' level")
            
            ticker_idx = sample.index.names.index('ticker')
            tickers = store.select(key, columns=[], start=0, stop=1000).index.get_level_values(ticker_idx).unique()
            log.info(f"Available tickers (first 10): {list(tickers)[:10]}")
            
            if self.ticker not in tickers:
                raise ValueError(f"Ticker {self.ticker} not found in the data")
            
            # Try multiple approaches to get the data
            df = None
            
            # Approach 1: Using where clause
            try:
                where_clause = f"ticker='{self.ticker}'"
                log.info(f"Loading data with where clause: {where_clause}")
                
                # Determine which columns to use
                columns = []
                if 'adj_close' in sample.columns:
                    columns.extend(['adj_close', 'adj_volume', 'adj_low', 'adj_high'])
                elif 'close' in sample.columns:
                    columns.extend(['close', 'volume', 'low', 'high'])
                else:
                    columns = list(sample.columns)[:4]
                log.info(f"Using columns: {columns}")
                
                # Try to get data with where clause
                df = store.select(key, where=where_clause, columns=columns)
                log.info(f"Loaded {df.shape[0]} rows for {self.ticker}")
            except Exception as e:
                log.warning(f"Error loading data with where clause: {e}")
                
                # Approach 2: Using IndexSlice
                try:
                    log.info("Trying alternative approach with IndexSlice")
                    idx = pd.IndexSlice
                    all_data = store.select(key, columns=columns)
                    df = all_data.loc[idx[:, self.ticker], :]
                    log.info(f"Loaded {df.shape[0]} rows with IndexSlice approach")
                except Exception as e2:
                    log.error(f"Error loading data with IndexSlice: {e2}")
                    raise ValueError(f"Could not load data for {self.ticker}: {e2}")
            
            # Check if we have enough data
            if df is None or df.empty or len(df) < self.trading_days:
                raise ValueError(f"Not enough data for {self.ticker}. Need at least {self.trading_days} rows, got {0 if df is None else len(df)}")
            
            # Rename columns for consistency
            column_mapping = {
                'adj_close': 'close',
                'adj_volume': 'volume',
                'adj_low': 'low',
                'adj_high': 'high'
            }
            df = df.rename(columns=column_mapping)
            
            # Keep only the columns we need
            df = df[['close', 'volume', 'low', 'high']]
            
            # Make sure the data is sorted by date
            df = df.sort_index(level='date')
            
            # Handle missing values
            if df.isna().any().any():
                missing_count = df.isna().sum().sum()
                log.warning(f"Data contains {missing_count} missing values. Filling with forward/backward fill.")
                df = df.ffill().bfill()
            
            log.info(f"Successfully loaded data for {self.ticker} with shape: {df.shape}")
            return df

    def preprocess_data(self):
        """Calculate technical indicators and returns, then remove missing values"""
        log.info('Preprocessing data for %s...', self.ticker)
        
        # Calculate returns at different timeframes
        self.data['returns'] = self.data.close.pct_change()
        self.data['ret_2'] = self.data.close.pct_change(2)
        self.data['ret_5'] = self.data.close.pct_change(5)
        self.data['ret_10'] = self.data.close.pct_change(10)
        self.data['ret_21'] = self.data.close.pct_change(21)
        
        # Calculate technical indicators
        try:
            self.data['rsi'] = talib.STOCHRSI(self.data.close)[1]
            self.data['macd'] = talib.MACD(self.data.close)[1]
            self.data['atr'] = talib.ATR(self.data.high, self.data.low, self.data.close)
            
            slowk, slowd = talib.STOCH(self.data.high, self.data.low, self.data.close)
            self.data['stoch'] = slowd - slowk
            self.data['ultosc'] = talib.ULTOSC(self.data.high, self.data.low, self.data.close)
        except Exception as e:
            log.warning(f"Error calculating technical indicators: {e}. Using simplified preprocessing.")
            # Simplified preprocessing if technical indicators fail
            self.data['rsi'] = np.nan
            self.data['macd'] = np.nan
            self.data['atr'] = np.nan
            self.data['stoch'] = np.nan
            self.data['ultosc'] = np.nan
        
        # Replace infinities and drop original price columns
        self.data = (self.data.replace((np.inf, -np.inf), np.nan)
                    .drop(['high', 'low', 'close', 'volume'], axis=1))
        
        # Handle missing values
        self.data = self.data.ffill().bfill()
        
        # Store returns separately before normalization
        r = self.data.returns.copy()
        
        # Normalize features if requested
        if self.normalize:
            try:
                self.data = pd.DataFrame(scale(self.data),
                                        columns=self.data.columns,
                                        index=self.data.index)
            except Exception as e:
                log.warning(f"Error scaling data: {e}. Skipping normalization.")
        
        # Get feature columns (all except returns)
        features = self.data.columns.drop('returns')
        
        # Create observation matrix
        self.features = self.data[features].values
        self.returns = r.values
        
        log.info(f"Preprocessed data shape: {self.data.shape}")
        
        # Verify we have enough data after preprocessing
        if len(self.data) < self.trading_days:
            raise ValueError(f"Not enough data after preprocessing. Need at least {self.trading_days} rows, got {len(self.data)}")
        self.data['returns'] = r  # don't scale returns
        self.data = self.data.loc[:, ['returns'] + list(features)]
        log.info(self.data.info())

    def reset(self):
        """Provides starting index for time series and resets step"""
        high = len(self.returns) - self.trading_days
        if high <= 0:
            raise ValueError(f"Not enough data for {self.trading_days} trading days after preprocessing")
            
        self.offset = np.random.randint(0, high)
        self.step = 0

    def take_step(self):
        """Returns data for current trading day and done signal"""
        if self.step < self.trading_days:
            observation = self.features[self.offset + self.step]
            self.step += 1
            return observation, self.step >= self.trading_days
        else:
            return self.features[self.offset + self.step - 1], True


class TradingSimulator:
    """Implements core trading simulator for single-instrument universe"""
    
    def __init__(self, steps, trading_cost_bps=1e-3, time_cost_bps=1e-4):
        self.trading_cost_bps = trading_cost_bps
        self.time_cost_bps = time_cost_bps
        self.steps = steps
        self.reset()

    def reset(self):
        self.step = 0
        self.actions = np.zeros(self.steps)
        self.navs = np.ones(self.steps)
        self.market_navs = np.ones(self.steps)
        self.strategy_returns = np.ones(self.steps)
        self.positions = np.zeros(self.steps)
        self.costs = np.zeros(self.steps)
        self.trades = np.zeros(self.steps)
        self.market_returns = np.zeros(self.steps)
        self.position = 0

    def take_step(self, action, market_return):
        """
        Calculates NAVs, trading costs and reward based on an action and latest market return
        and returns the reward and a summary of the day's activity
        """
        # Update market return and step
        self.market_returns[self.step] = market_return
        
        # Calculate transaction and holding costs
        start_position = self.position
        start_nav = self.navs[max(0, self.step - 1)]
        start_market_nav = self.market_navs[max(0, self.step - 1)]
        
        # Decode action and update position
        self.actions[self.step] = action
        self.positions[self.step] = action - 1  # Convert action (0, 1, 2) to position (-1, 0, 1)
        self.position = self.positions[self.step]
        
        # Calculate costs and trades
        n_trades = abs(self.position - start_position)
        self.trades[self.step] = n_trades
        
        # Calculate costs and reward
        trade_costs = abs(n_trades) * self.trading_cost_bps
        time_cost = 0 if n_trades else self.time_cost_bps
        self.costs[self.step] = trade_costs + time_cost
        reward = start_position * market_return - self.costs[max(0, self.step-1)]
        self.strategy_returns[self.step] = reward
        
        # Update NAVs
        if self.step != 0:
            self.navs[self.step] = start_nav * (1 + self.strategy_returns[self.step])
            self.market_navs[self.step] = start_market_nav * (1 + self.market_returns[self.step])
        
        # Prepare info dictionary
        info = {
            'reward': reward,
            'nav': self.navs[self.step],
            'costs': self.costs[self.step]
        }
        
        self.step += 1
        return reward, info

    def result(self):
        """returns current state as pd.DataFrame """
        return pd.DataFrame({'action'         : self.actions,  # current action
                             'nav'            : self.navs,  # starting Net Asset Value (NAV)
                             'market_nav'     : self.market_navs,
                             'market_return'  : self.market_returns,
                             'strategy_return': self.strategy_returns,
                             'position'       : self.positions,  # eod position
                             'cost'           : self.costs,  # eod costs
                             'trade'          : self.trades})  # eod trade)


class TradingEnvironment(gym.Env):
    """
    A trading environment for reinforcement learning.
    
    Provides daily observations for a stock price series
    An episode is defined as a sequence of 252 trading days with random start
    Each day is a 'step' that allows the agent to choose one of three actions:
    - 0: SHORT
    - 1: HOLD
    - 2: LONG
    
    Trading has an optional cost (default: 10bps) of the change in position value.
    Going from short to long implies two trades.
    Not trading also incurs a default time cost of 1bps per step.
    
    This version is updated to be compatible with Gym API v0.26+ and
    only uses real data from the HDF5 file.
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, trading_days=252, trading_cost_bps=1e-3,
                 time_cost_bps=1e-4, ticker='AAPL'):
        self.trading_days = trading_days
        self.trading_cost_bps = trading_cost_bps
        self.ticker = ticker
        self.time_cost_bps = time_cost_bps
        
        # Initialize data source and simulator
        self.data_source = DataSource(
            trading_days=self.trading_days,
            ticker=self.ticker
        )
        
        self.simulator = TradingSimulator(
            steps=self.trading_days,
            trading_cost_bps=self.trading_cost_bps,
            time_cost_bps=self.time_cost_bps
        )
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)
        
        # Get the shape of the observation from the features
        obs_dim = self.data_source.features.shape[1]
        
        # Create observation space with correct dimensions
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Initialize environment
        self.reset()
    
    def seed(self, seed=None):
        """Set the random seed"""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """Returns state observation, reward, terminated, truncated, and info"""
        assert self.action_space.contains(action), f'{action} {type(action)} invalid'
        
        # Get observation and done signal from data source
        observation, done = self.data_source.take_step()
        
        # Get reward and info from simulator
        reward, info = self.simulator.take_step(
            action=action,
            market_return=self.data_source.returns[self.data_source.offset + self.data_source.step - 1]
        )
        
        # For gym v26+, we need to return terminated and truncated separately
        return observation.astype(np.float32), reward, done, False, info

    def reset(self, seed=None, options=None):
        """Resets DataSource and TradingSimulator; returns first observation and info"""
        if seed is not None:
            self.seed(seed)
        
        # Reset data source and simulator
        self.data_source.reset()
        self.simulator.reset()
        
        # Get first observation
        observation, _ = self.data_source.take_step()
        
        # For gym v26+, we need to return observation and info
        return observation.astype(np.float32), {}
        
    def render(self, mode='human'):
        """Not implemented"""
        pass


def setup_trading_env(trading_days=252):
    """
    Register the trading environment and return a function to create it
    
    Parameters:
    -----------
    trading_days : int
        Number of trading days in an episode
        
    Returns:
    --------
    create_env : function
        Function to create the trading environment
    """
    # Check if already registered and remove if exists
    try:
        gym.envs.registry.pop('trading-v0')
        print("Removed existing trading-v0 registration")
    except (KeyError, ValueError):
        pass
    
    # Register the environment
    register(
        id='trading-v0',
        entry_point='src.backtesting.trading_env:TradingEnvironment',
        max_episode_steps=trading_days
    )
    print("Trading environment registered successfully")
    
    # Return a function to create the environment
    def create_env(ticker='AAPL', trading_cost_bps=1e-3, time_cost_bps=1e-4):
        return gym.make('trading-v0',
                       ticker=ticker,
                       trading_days=trading_days,
                       trading_cost_bps=trading_cost_bps,
                       time_cost_bps=time_cost_bps)
    
    return create_env


# Example usage
if __name__ == "__main__":
    # Register and get the environment creation function
    create_env = setup_trading_env(trading_days=252)
    
    # Create the environment with a ticker that exists in your dataset
    env = create_env(ticker='AAPL', trading_cost_bps=1e-3, time_cost_bps=1e-4)
    
    # Test the environment
    print("Testing environment...")
    try:
        obs, info = env.reset(seed=42)
        print(f"Observation shape: {obs.shape}")
        
        # Take a step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step result - reward: {reward}, terminated: {terminated}, truncated: {truncated}")
        print("Environment test successful!")
    except Exception as e:
        print(f"Environment test failed: {e}")

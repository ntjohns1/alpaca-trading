"""
Backtesting framework for RL trading strategies
Builds on the trading_env.py implementation from the notebooks
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path

# Import the trading environment components
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import the trading environment components
from notebooks.trading_env import DataSource, TradingSimulator, TradingEnvironment

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class BacktestEngine:
    """
    Enhanced backtesting framework for RL trading strategies
    
    Extends the functionality of the TradingEnvironment to:
    - Run multiple episodes with different starting points
    - Compare multiple strategies
    - Generate performance metrics and visualizations
    - Save results for further analysis
    """
    
    def __init__(self, 
                 ticker='AAPL',
                 trading_days=252,
                 trading_cost_bps=1e-3,
                 time_cost_bps=1e-4,
                 data_dir=None):
        """
        Initialize the backtest engine
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        trading_days : int
            Number of trading days per episode
        trading_cost_bps : float
            Trading cost in basis points
        time_cost_bps : float
            Time cost in basis points
        data_dir : str or Path
            Directory containing historical data
        """
        self.ticker = ticker
        self.trading_days = trading_days
        self.trading_cost_bps = trading_cost_bps
        self.time_cost_bps = time_cost_bps
        self.data_dir = Path(data_dir) if data_dir else None
        
        # Create the trading environment
        self.env = TradingEnvironment(
            ticker=ticker,
            trading_days=trading_days,
            trading_cost_bps=trading_cost_bps,
            time_cost_bps=time_cost_bps
        )
        
        # Results storage
        self.results = {}
        self.episode_history = []
        
    def run_episode(self, agent, episode_idx=0, render=False):
        """
        Run a single backtest episode
        
        Parameters:
        -----------
        agent : object
            Trading agent with predict method
        episode_idx : int
            Episode index for tracking
        render : bool
            Whether to render the environment
            
        Returns:
        --------
        dict : Episode results
        """
        # Reset the environment
        state = self.env.reset()
        done = False
        total_reward = 0
        
        # Track actions and states
        actions = []
        states = []
        rewards = []
        
        # Run until episode is done
        while not done:
            # Get action from agent
            action = agent.predict(state)
            
            # Take action in environment
            next_state, reward, done, info = self.env.step(action)
            
            # Store results
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            # Update state and reward
            state = next_state
            total_reward += reward
            
            # Render if requested
            if render:
                self.env.render()
        
        # Get final portfolio value
        portfolio_value = self.env.simulator.navs[-1]
        market_value = self.env.simulator.market_navs[-1]
        
        # Calculate metrics
        sharpe = self._calculate_sharpe(rewards)
        max_drawdown = self._calculate_max_drawdown(self.env.simulator.navs)
        
        # Store episode results
        episode_results = {
            'episode': episode_idx,
            'total_reward': total_reward,
            'portfolio_value': portfolio_value,
            'market_value': market_value,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'outperformance': portfolio_value - market_value,
            'actions': actions,
            'states': states,
            'rewards': rewards
        }
        
        self.episode_history.append(episode_results)
        return episode_results
    
    def run_backtest(self, agent, episodes=100, verbose=True):
        """
        Run multiple backtest episodes
        
        Parameters:
        -----------
        agent : object
            Trading agent with predict method
        episodes : int
            Number of episodes to run
        verbose : bool
            Whether to print progress
            
        Returns:
        --------
        dict : Aggregated backtest results
        """
        # Reset results
        self.episode_history = []
        
        # Run episodes
        for i in range(episodes):
            if verbose and (i % 10 == 0):
                log.info(f"Running episode {i+1}/{episodes}")
            
            # Run episode
            self.run_episode(agent, episode_idx=i)
        
        # Aggregate results
        self._aggregate_results()
        
        if verbose:
            self._print_summary()
            
        return self.results
    
    def _aggregate_results(self):
        """Aggregate results from all episodes"""
        if not self.episode_history:
            return
        
        # Extract key metrics
        portfolio_values = [ep['portfolio_value'] for ep in self.episode_history]
        market_values = [ep['market_value'] for ep in self.episode_history]
        total_rewards = [ep['total_reward'] for ep in self.episode_history]
        sharpe_ratios = [ep['sharpe_ratio'] for ep in self.episode_history]
        max_drawdowns = [ep['max_drawdown'] for ep in self.episode_history]
        outperformances = [ep['outperformance'] for ep in self.episode_history]
        
        # Calculate win rate (outperformance > 0)
        win_rate = sum(1 for o in outperformances if o > 0) / len(outperformances)
        
        # Store aggregated results
        self.results = {
            'episodes': len(self.episode_history),
            'mean_portfolio_value': np.mean(portfolio_values),
            'mean_market_value': np.mean(market_values),
            'mean_reward': np.mean(total_rewards),
            'mean_sharpe': np.mean(sharpe_ratios),
            'mean_max_drawdown': np.mean(max_drawdowns),
            'mean_outperformance': np.mean(outperformances),
            'win_rate': win_rate,
            'portfolio_values': portfolio_values,
            'market_values': market_values,
            'outperformances': outperformances
        }
    
    def _calculate_sharpe(self, rewards):
        """Calculate Sharpe ratio from rewards"""
        if not rewards:
            return 0
        
        returns = np.array(rewards)
        sharpe = np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252)  # Annualized
        return sharpe
    
    def _calculate_max_drawdown(self, values):
        """Calculate maximum drawdown from portfolio values"""
        if not values:
            return 0
            
        values = np.array(values)
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        max_drawdown = np.min(drawdown)
        return max_drawdown
    
    def _print_summary(self):
        """Print summary of backtest results"""
        if not self.results:
            log.info("No results to display")
            return
            
        log.info(f"Backtest Summary ({self.results['episodes']} episodes):")
        log.info(f"Mean Portfolio Value: {self.results['mean_portfolio_value']:.4f}")
        log.info(f"Mean Market Value: {self.results['mean_market_value']:.4f}")
        log.info(f"Mean Outperformance: {self.results['mean_outperformance']:.4f}")
        log.info(f"Win Rate: {self.results['win_rate']:.2%}")
        log.info(f"Mean Sharpe Ratio: {self.results['mean_sharpe']:.4f}")
        log.info(f"Mean Max Drawdown: {self.results['mean_max_drawdown']:.2%}")
    
    def plot_results(self, save_path=None):
        """
        Plot backtest results
        
        Parameters:
        -----------
        save_path : str or Path
            Path to save the plot
        """
        if not self.results:
            log.info("No results to plot")
            return
            
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot portfolio value distribution
        axes[0].hist(self.results['portfolio_values'], bins=20, alpha=0.5, label='Strategy')
        axes[0].hist(self.results['market_values'], bins=20, alpha=0.5, label='Market')
        axes[0].set_title('Distribution of Final Portfolio Values')
        axes[0].set_xlabel('Portfolio Value')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        
        # Plot outperformance distribution
        axes[1].hist(self.results['outperformances'], bins=20)
        axes[1].axvline(0, color='red', linestyle='--')
        axes[1].set_title('Distribution of Strategy Outperformance')
        axes[1].set_xlabel('Outperformance')
        axes[1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            log.info(f"Plot saved to {save_path}")
        
        plt.show()
        
    def save_results(self, path):
        """
        Save backtest results to CSV
        
        Parameters:
        -----------
        path : str or Path
            Path to save the results
        """
        if not self.results:
            log.info("No results to save")
            return
            
        # Create DataFrame from episode history
        df = pd.DataFrame([
            {
                'episode': ep['episode'],
                'portfolio_value': ep['portfolio_value'],
                'market_value': ep['market_value'],
                'outperformance': ep['outperformance'],
                'sharpe_ratio': ep['sharpe_ratio'],
                'max_drawdown': ep['max_drawdown'],
                'total_reward': ep['total_reward']
            }
            for ep in self.episode_history
        ])
        
        # Save to CSV
        df.to_csv(path, index=False)
        log.info(f"Results saved to {path}")

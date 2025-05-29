"""
Example script demonstrating how to use the backtesting framework
and live trading interface with a DQN agent
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import components
from src.backtesting.backtest_engine import BacktestEngine
from src.live_trading.alpaca_interface import AlpacaInterface
from src.agents.dqn_agent import DQNAgent

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def train_agent_with_backtest(ticker='AAPL', episodes=100, state_size=10):
    """
    Train a DQN agent using the backtesting framework
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    episodes : int
        Number of episodes to train
    state_size : int
        Size of the state vector
        
    Returns:
    --------
    DQNAgent : Trained agent
    """
    log.info(f"Training agent on {ticker} for {episodes} episodes")
    
    # Initialize agent
    agent = DQNAgent(state_size=state_size)
    
    # Initialize backtest engine
    engine = BacktestEngine(ticker=ticker)
    
    # Training loop
    for episode in range(episodes):
        if episode % 10 == 0:
            log.info(f"Episode {episode+1}/{episodes}")
        
        # Reset environment
        state = engine.env.reset()
        done = False
        total_reward = 0
        
        # Run episode
        while not done:
            # Choose action
            action = agent.act(state)
            
            # Take action
            next_state, reward, done, _ = engine.env.step(action)
            
            # Remember experience
            agent.remember(state, action, reward, next_state, done)
            
            # Update state and reward
            state = next_state
            total_reward += reward
            
            # Train agent
            agent.replay()
        
        # Update target network every 10 episodes
        if episode % 10 == 0:
            agent.update_target_model()
            
        # Log progress
        if episode % 10 == 0:
            log.info(f"Episode {episode+1}: Total reward: {total_reward:.4f}, Epsilon: {agent.epsilon:.4f}")
    
    # Run backtest with trained agent
    results = engine.run_backtest(agent, episodes=10, verbose=True)
    
    # Plot results
    engine.plot_results()
    
    # Save agent
    models_dir = Path(__file__).parent.parent.parent / 'models'
    models_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    agent.save(str(models_dir / f"dqn_agent_{ticker}_{timestamp}.h5"))
    
    log.info(f"Agent trained and saved to models/dqn_agent_{ticker}_{timestamp}.h5")
    
    return agent

def live_trade_with_agent(agent, symbols=['AAPL'], interval_seconds=60):
    """
    Trade live with a trained DQN agent
    
    Parameters:
    -----------
    agent : DQNAgent
        Trained agent
    symbols : list
        List of stock symbols to trade
    interval_seconds : int
        Seconds between trading decisions
    """
    log.info(f"Starting live trading with agent on symbols: {symbols}")
    
    # Initialize Alpaca interface
    alpaca = AlpacaInterface(
        max_position_value=10000,  # $10k max position
        max_single_order_value=1000,  # $1k per order
        max_daily_drawdown_pct=5.0,  # 5% max daily drawdown
        max_total_drawdown_pct=10.0,  # 10% max total drawdown
        paper_trading=True  # Use paper trading
    )
    
    # Check account
    account_info = alpaca.get_account_info()
    log.info(f"Account equity: ${account_info['equity']:.2f}")
    
    try:
        # Main trading loop
        while True:
            # Check if trading is allowed
            if not alpaca.is_trading_allowed:
                log.warning("Trading is not allowed - circuit breaker triggered")
                break
                
            # Get state features for each symbol
            features = alpaca.get_state_features(symbols)
            
            for symbol in symbols:
                if symbol not in features:
                    log.warning(f"No features available for {symbol}")
                    continue
                
                # Convert features to state vector
                state = agent.get_state_from_features(features[symbol])
                
                # Get action from agent
                action = agent.predict(state)
                
                # Execute action
                result = alpaca.execute_action(symbol, action, value=1000)  # $1k per trade
                
                if result['success']:
                    log.info(f"Order for {symbol}: {result}")
                else:
                    log.warning(f"Order failed for {symbol}: {result}")
            
            # Wait for next interval
            log.info(f"Waiting {interval_seconds} seconds until next trading decision...")
            import time
            time.sleep(interval_seconds)
            
    except KeyboardInterrupt:
        log.info("Trading stopped by user")
    except Exception as e:
        log.error(f"Error during live trading: {e}")
    finally:
        # Cancel all open orders
        alpaca.cancel_all_orders()
        log.info("All open orders canceled")

def main():
    """Main function"""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Backtest and trade with DQN agent')
    parser.add_argument('--mode', type=str, default='backtest', choices=['backtest', 'train', 'trade'],
                        help='Mode to run: backtest, train, or trade')
    parser.add_argument('--ticker', type=str, default='AAPL',
                        help='Stock ticker symbol')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of episodes for training')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model weights for trading')
    args = parser.parse_args()
    
    if args.mode == 'backtest':
        # Run backtest with pre-trained agent
        if args.model is None:
            log.error("Model path is required for backtest mode")
            return
            
        # Load agent
        agent = DQNAgent(state_size=6)  # Adjust state size as needed
        agent.load(args.model)
        
        # Run backtest
        engine = BacktestEngine(ticker=args.ticker)
        results = engine.run_backtest(agent, episodes=10, verbose=True)
        engine.plot_results()
        
    elif args.mode == 'train':
        # Train agent with backtest
        train_agent_with_backtest(ticker=args.ticker, episodes=args.episodes)
        
    elif args.mode == 'trade':
        # Trade live with trained agent
        if args.model is None:
            log.error("Model path is required for trade mode")
            return
            
        # Load agent
        agent = DQNAgent(state_size=6)  # Adjust state size as needed
        agent.load(args.model)
        
        # Run live trading
        live_trade_with_agent(agent, symbols=[args.ticker])

if __name__ == "__main__":
    main()

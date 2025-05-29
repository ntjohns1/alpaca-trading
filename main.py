#!/usr/bin/env python
"""
Alpaca Trading with Reinforcement Learning
Main script for backtesting and live trading

This script provides a command-line interface for:
1. Running backtests with the RL agent
2. Training the RL agent using historical data
3. Running live trading with a trained agent
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# Import components
from src.backtesting.backtest_engine import BacktestEngine
from src.live_trading.alpaca_interface import AlpacaInterface
from src.agents.dqn_agent import DQNAgent

def backtest(args):
    """Run backtesting with a trained or new agent"""
    log.info(f"Running backtest on {args.ticker} for {args.episodes} episodes")
    
    # Initialize backtest engine
    engine = BacktestEngine(
        ticker=args.ticker,
        trading_days=args.trading_days,
        trading_cost_bps=args.trading_cost_bps,
        time_cost_bps=args.time_cost_bps
    )
    
    # Initialize agent
    state_size = args.state_size
    agent = DQNAgent(state_size=state_size)
    
    # Load model if specified
    if args.model:
        log.info(f"Loading model from {args.model}")
        agent.load(args.model)
    
    # Run backtest
    results = engine.run_backtest(agent, episodes=args.episodes, verbose=True)
    
    # Plot and save results
    if args.plot:
        engine.plot_results()
    
    if args.save_results:
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        engine.save_results(results_dir / f"backtest_{args.ticker}_{timestamp}.csv")
    
    return results

def train(args):
    """Train an agent using the backtesting framework"""
    log.info(f"Training agent on {args.ticker} for {args.episodes} episodes")
    
    # Initialize agent
    state_size = args.state_size
    agent = DQNAgent(
        state_size=state_size,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size
    )
    
    # Load model if continuing training
    if args.model:
        log.info(f"Loading model from {args.model} to continue training")
        agent.load(args.model)
    
    # Initialize backtest engine
    engine = BacktestEngine(
        ticker=args.ticker,
        trading_days=args.trading_days,
        trading_cost_bps=args.trading_cost_bps,
        time_cost_bps=args.time_cost_bps
    )
    
    # Training loop
    for episode in range(args.episodes):
        if episode % 10 == 0:
            log.info(f"Episode {episode+1}/{args.episodes}")
        
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
    if args.plot:
        engine.plot_results()
    
    # Save agent
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = str(models_dir / f"dqn_agent_{args.ticker}_{timestamp}.h5")
    agent.save(model_path)
    
    log.info(f"Agent trained and saved to {model_path}")
    
    return agent

def trade(args):
    """Run live trading with a trained agent"""
    if not args.model:
        log.error("Model path is required for trade mode")
        return
    
    log.info(f"Starting live trading with agent on symbols: {args.symbols}")
    
    # Load agent
    agent = DQNAgent(state_size=args.state_size)
    agent.load(args.model)
    
    # Initialize Alpaca interface
    alpaca = AlpacaInterface(
        max_position_value=args.max_position_value,
        max_single_order_value=args.max_order_value,
        max_daily_drawdown_pct=args.max_daily_drawdown,
        max_total_drawdown_pct=args.max_total_drawdown,
        paper_trading=not args.live
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
            features = alpaca.get_state_features(args.symbols)
            
            for symbol in args.symbols:
                if symbol not in features:
                    log.warning(f"No features available for {symbol}")
                    continue
                
                # Convert features to state vector
                state = agent.get_state_from_features(features[symbol])
                
                # Get action from agent
                action = agent.predict(state)
                
                # Execute action
                result = alpaca.execute_action(symbol, action, value=args.order_value)
                
                if result['success']:
                    log.info(f"Order for {symbol}: {result}")
                else:
                    log.warning(f"Order failed for {symbol}: {result}")
            
            # Wait for next interval
            log.info(f"Waiting {args.interval} seconds until next trading decision...")
            import time
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        log.info("Trading stopped by user")
    except Exception as e:
        log.error(f"Error during live trading: {e}")
    finally:
        # Cancel all open orders
        alpaca.cancel_all_orders()
        log.info("All open orders canceled")
        
        if args.liquidate_on_exit:
            log.info("Liquidating all positions")
            alpaca.liquidate_all_positions()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Alpaca Trading with Reinforcement Learning')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker symbol')
    common_parser.add_argument('--state-size', type=int, default=6, help='Size of the state vector')
    common_parser.add_argument('--model', type=str, help='Path to model weights file')
    common_parser.add_argument('--plot', action='store_true', help='Plot results')
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', parents=[common_parser], help='Run backtesting')
    backtest_parser.add_argument('--episodes', type=int, default=100, help='Number of episodes to run')
    backtest_parser.add_argument('--trading-days', type=int, default=252, help='Number of trading days per episode')
    backtest_parser.add_argument('--trading-cost-bps', type=float, default=1e-3, help='Trading cost in basis points')
    backtest_parser.add_argument('--time-cost-bps', type=float, default=1e-4, help='Time cost in basis points')
    backtest_parser.add_argument('--save-results', action='store_true', help='Save backtest results to CSV')
    
    # Train command
    train_parser = subparsers.add_parser('train', parents=[common_parser], help='Train agent with backtesting')
    train_parser.add_argument('--episodes', type=int, default=100, help='Number of episodes to train')
    train_parser.add_argument('--trading-days', type=int, default=252, help='Number of trading days per episode')
    train_parser.add_argument('--trading-cost-bps', type=float, default=1e-3, help='Trading cost in basis points')
    train_parser.add_argument('--time-cost-bps', type=float, default=1e-4, help='Time cost in basis points')
    train_parser.add_argument('--gamma', type=float, default=0.95, help='Discount factor')
    train_parser.add_argument('--epsilon', type=float, default=1.0, help='Initial exploration rate')
    train_parser.add_argument('--epsilon-min', type=float, default=0.01, help='Minimum exploration rate')
    train_parser.add_argument('--epsilon-decay', type=float, default=0.995, help='Exploration decay rate')
    train_parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    
    # Trade command
    trade_parser = subparsers.add_parser('trade', parents=[common_parser], help='Run live trading')
    trade_parser.add_argument('--symbols', type=str, nargs='+', default=['AAPL'], help='Stock symbols to trade')
    trade_parser.add_argument('--interval', type=int, default=60, help='Seconds between trading decisions')
    trade_parser.add_argument('--max-position-value', type=float, default=10000, help='Maximum position value in dollars')
    trade_parser.add_argument('--max-order-value', type=float, default=1000, help='Maximum single order value in dollars')
    trade_parser.add_argument('--order-value', type=float, default=1000, help='Value per order in dollars')
    trade_parser.add_argument('--max-daily-drawdown', type=float, default=5.0, help='Maximum daily drawdown percentage')
    trade_parser.add_argument('--max-total-drawdown', type=float, default=10.0, help='Maximum total drawdown percentage')
    trade_parser.add_argument('--live', action='store_true', help='Use live trading instead of paper trading')
    trade_parser.add_argument('--liquidate-on-exit', action='store_true', help='Liquidate all positions on exit')
    
    args = parser.parse_args()
    
    if args.command == 'backtest':
        backtest(args)
    elif args.command == 'train':
        train(args)
    elif args.command == 'trade':
        trade(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

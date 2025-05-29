# Alpaca Trading with Reinforcement Learning

This project implements a reinforcement learning (RL) framework for algorithmic trading using the Alpaca API. It consists of a backtesting framework for training and evaluating RL agents, and an interface for deploying trained agents for live trading.

## Project Structure

```
alpaca-trading/
├── data/                  # Directory for market data
├── models/                # Directory for saved model weights
├── notebooks/             # Jupyter notebooks for analysis and exploration
├── src/
│   ├── agents/            # RL agent implementations
│   │   └── dqn_agent.py   # Deep Q-Network agent
│   ├── backtesting/       # Backtesting framework
│   │   └── backtest_engine.py  # Enhanced backtesting engine
│   ├── data/              # Data processing modules
│   │   └── data_processor.py   # ETL pipeline for market data
│   ├── examples/          # Example usage scripts
│   │   └── backtest_and_trade.py  # Example of backtesting and trading
│   ├── live_trading/      # Live trading components
│   │   └── alpaca_interface.py  # Interface to Alpaca API with safety features
│   └── alpaca_client.py   # Basic Alpaca API client
├── data_etl.py            # Script for data extraction, transformation, and loading
├── main.py                # Main script for backtesting and live trading
└── requirements.txt       # Project dependencies
```

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your Alpaca API credentials:
   ```
   ALPACA_API_KEY=your_api_key
   ALPACA_API_SECRET=your_api_secret
   ALPACA_PAPER=True  # Set to False for live trading
   ```

## Data Preparation

Before backtesting or training, you need to download and process market data:

```bash
python data_etl.py --symbols AAPL MSFT GOOGL --timeframe day --limit 1000 --output market_data.h5
```

Options:
- `--symbols`: List of stock symbols to download data for
- `--timeframe`: Timeframe for data (day, hour, 15min, 5min, 1min)
- `--start-date`: Start date in YYYY-MM-DD format
- `--end-date`: End date in YYYY-MM-DD format (default: today)
- `--limit`: Number of bars to download (used if start-date not provided)
- `--output`: Output HDF5 filename
- `--data-dir`: Directory to store data files

## Usage

The main script provides a command-line interface for backtesting, training, and live trading:

### Backtesting

Test a trained agent on historical data:

```bash
python main.py backtest --ticker AAPL --model models/dqn_agent_AAPL.h5 --episodes 100 --plot
```

Options:
- `--ticker`: Stock ticker symbol
- `--model`: Path to trained model weights
- `--episodes`: Number of episodes to run
- `--trading-days`: Number of trading days per episode
- `--trading-cost-bps`: Trading cost in basis points
- `--time-cost-bps`: Time cost in basis points
- `--plot`: Plot results
- `--save-results`: Save backtest results to CSV

### Training

Train a DQN agent using the backtesting framework:

```bash
python main.py train --ticker AAPL --episodes 500 --plot
```

Options:
- `--ticker`: Stock ticker symbol
- `--episodes`: Number of episodes to train
- `--gamma`: Discount factor
- `--epsilon`: Initial exploration rate
- `--epsilon-min`: Minimum exploration rate
- `--epsilon-decay`: Exploration decay rate
- `--learning-rate`: Learning rate
- `--batch-size`: Batch size for training
- `--plot`: Plot results

### Live Trading

Deploy a trained agent for live trading:

```bash
python main.py trade --model models/dqn_agent_AAPL.h5 --symbols AAPL --interval 60
```

Options:
- `--model`: Path to trained model weights
- `--symbols`: Stock symbols to trade
- `--interval`: Seconds between trading decisions
- `--max-position-value`: Maximum position value in dollars
- `--max-order-value`: Maximum single order value in dollars
- `--order-value`: Value per order in dollars
- `--max-daily-drawdown`: Maximum daily drawdown percentage
- `--max-total-drawdown`: Maximum total drawdown percentage
- `--live`: Use live trading instead of paper trading
- `--liquidate-on-exit`: Liquidate all positions on exit

## Components

### Backtesting Framework

The backtesting framework builds on the trading environment from the notebooks and provides:
- Metrics like Sharpe ratio, max drawdown, and win rate
- Multiple episodes with different starting points
- Visualization and result saving capabilities

### Alpaca Interface for Live Trading

The Alpaca interface provides:
- Connection to Alpaca API for executing trades
- Safety features like circuit breakers and position limits
- Methods for getting account info, positions, and historical data
- Conversion of RL agent actions to actual orders

### DQN Agent

The DQN agent implementation features:
- Experience replay
- Target networks
- Double DQN
- Compatibility with both backtesting and live trading

## Safety Features

The live trading interface includes several safety features:
- Maximum position value limits
- Maximum single order value limits
- Daily and total drawdown circuit breakers
- Position monitoring and risk management

## Example Workflow

1. Download and process historical data:
   ```bash
   python data_etl.py --symbols AAPL --timeframe day --limit 1000
   ```

2. Train an agent using the backtesting framework:
   ```bash
   python main.py train --ticker AAPL --episodes 500 --plot
   ```

3. Evaluate the trained agent with backtesting:
   ```bash
   python main.py backtest --ticker AAPL --model models/dqn_agent_AAPL_*.h5 --episodes 100 --plot --save-results
   ```

4. Deploy the agent for paper trading:
   ```bash
   python main.py trade --model models/dqn_agent_AAPL_*.h5 --symbols AAPL --interval 60
   ```

5. When satisfied with performance, deploy for live trading:
   ```bash
   python main.py trade --model models/dqn_agent_AAPL_*.h5 --symbols AAPL --interval 60 --live
   ```

## License

MIT License

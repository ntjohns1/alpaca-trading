#!/usr/bin/env python
"""
Data ETL Script for Alpaca Trading

This script provides a command-line interface for:
1. Downloading historical market data from Alpaca
2. Processing and calculating features
3. Storing the data in HDF5 files for efficient access by the backtesting framework
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from alpaca.data.timeframe import TimeFrame

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

# Import components
from src.data.data_processor import DataProcessor

def parse_date(date_str):
    """Parse date string in YYYY-MM-DD format"""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        log.error(f"Invalid date format: {date_str}. Use YYYY-MM-DD.")
        sys.exit(1)

def parse_timeframe(timeframe_str):
    """Parse timeframe string to TimeFrame enum"""
    timeframe_map = {
        'day': TimeFrame.Day,
        'hour': TimeFrame.Hour,
        '1hour': TimeFrame.Hour,
        '15min': TimeFrame.Minute,
        '5min': TimeFrame.Minute,
        '1min': TimeFrame.Minute
    }
    
    if timeframe_str.lower() not in timeframe_map:
        log.error(f"Invalid timeframe: {timeframe_str}. Use day, hour, 15min, 5min, or 1min.")
        sys.exit(1)
        
    return timeframe_map[timeframe_str.lower()]

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Data ETL for Alpaca Trading')
    
    parser.add_argument('--symbols', type=str, nargs='+', required=True,
                        help='Stock symbols to download data for')
    parser.add_argument('--timeframe', type=str, default='day',
                        help='Timeframe for data: day, hour, 15min, 5min, 1min')
    parser.add_argument('--start-date', type=str,
                        help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end-date', type=str,
                        help='End date in YYYY-MM-DD format (default: today)')
    parser.add_argument('--limit', type=int, default=1000,
                        help='Number of bars to download (used if start-date not provided)')
    parser.add_argument('--output', type=str, default='market_data.h5',
                        help='Output HDF5 filename')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory to store data files')
    
    args = parser.parse_args()
    
    # Parse dates
    start_date = parse_date(args.start_date)
    end_date = parse_date(args.end_date)
    
    # Parse timeframe
    timeframe = parse_timeframe(args.timeframe)
    
    # Initialize data processor
    data_dir = Path(args.data_dir)
    data_dir.mkdir(exist_ok=True)
    
    processor = DataProcessor(data_dir=data_dir)
    
    # Run ETL pipeline
    log.info(f"Running ETL pipeline for symbols: {args.symbols}")
    log.info(f"Timeframe: {args.timeframe}, Output: {args.output}")
    
    processed_data = processor.run_etl_pipeline(
        symbols=args.symbols,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        limit=args.limit,
        filename=args.output
    )
    
    # Print summary
    log.info("ETL pipeline completed")
    log.info(f"Data saved to: {data_dir / args.output}")
    
    for symbol, df in processed_data.items():
        log.info(f"{symbol}: {len(df)} rows, {df.index.min()} to {df.index.max()}")

if __name__ == "__main__":
    main()

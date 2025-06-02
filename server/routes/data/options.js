import express from 'express';
import { alpaca } from '../../client/index.js';

const router = express.Router();

// Get option bars for specific symbols
router.get('/bars', async (req, res) => {
  try {
    const { symbols, start, end, limit, timeframe } = req.query;

    const options = {
      start: start ? new Date(start) : undefined,
      end: end ? new Date(end) : undefined,
      timeframe: timeframe || '1Min',  // Default timeframe
      limit: limit ? parseInt(limit) : 100
    };

    const symbolsArray = symbols.split(',');
    const bars = await alpaca.getOptionBars(symbolsArray, options);
    res.json(Object.fromEntries(bars));
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to fetch option bars' });
  }
});

// Get option trades for specific symbols
router.get('/trades', async (req, res) => {
  try {
    const { symbols, start, end, limit } = req.query;

    const options = {
      start: start ? new Date(start) : undefined,
      end: end ? new Date(end) : undefined,
      limit: limit ? parseInt(limit) : 100
    };

    const symbolsArray = symbols.split(',');
    const trades = await alpaca.getOptionTrades(symbolsArray, options);
    res.json(Object.fromEntries(trades));
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to fetch option trades' });
  }
});

// Get latest option trades for specific symbols
router.get('/trades/latest', async (req, res) => {
  try {
    const { symbols } = req.query;
    const symbolsArray = symbols.split(',');

    const latestTrades = await alpaca.getOptionLatestTrades(symbolsArray);
    res.json(Object.fromEntries(latestTrades));
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to fetch latest option trades' });
  }
});

// Get latest option quotes for specific symbols
router.get('/quotes/latest', async (req, res) => {
  try {
    const { symbols } = req.query;
    const symbolsArray = symbols.split(',');

    const latestQuotes = await alpaca.getOptionLatestQuotes(symbolsArray);
    res.json(Object.fromEntries(latestQuotes));
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to fetch latest option quotes' });
  }
});

// Get option snapshots for specific symbols
router.get('/snapshots', async (req, res) => {
  try {
    const { symbols } = req.query;
    const symbolsArray = symbols.split(',');

    const snapshots = await alpaca.getOptionSnapshots(symbolsArray);
    res.json(snapshots);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to fetch option snapshots' });
  }
});

// Get option chain for an underlying symbol
router.get('/chain', async (req, res) => {
  try {
    const { underlyingSymbol, expiry, limit } = req.query;

    const options = {
      expiry: expiry ? new Date(expiry) : undefined,
      limit: limit ? parseInt(limit) : 100
    };

    const optionChain = await alpaca.getOptionChain(underlyingSymbol, options);
    res.json(optionChain);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to fetch option chain' });
  }
});

export default router;
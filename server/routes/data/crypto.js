import express from 'express';
import { alpaca } from '../../client/index.js';

const router = express.Router();

// Get crypto trades for specific symbols
router.get('/trades', async (req, res) => {
  try {
    const { symbols, start, end, limit } = req.query;

    const options = {
      start: start ? new Date(start) : undefined,
      end: end ? new Date(end) : undefined,
      limit: limit ? parseInt(limit) : 100
    };

    const symbolsArray = symbols.split(',');
    const trades = alpaca.getCryptoTrades(symbolsArray, options);
    const result = [];

    for await (let trade of trades) {
      result.push(trade);
    }

    res.json(result);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to fetch crypto trades' });
  }
});

// Get latest crypto trades
router.get('/trades/latest', async (req, res) => {
  try {
    const { symbols } = req.query;
    const symbolsArray = symbols.split(',');

    const trades = await alpaca.getLatestCryptoTrades(symbolsArray);
    res.json(Object.fromEntries(trades));
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to fetch latest crypto trades' });
  }
});

// Get crypto quotes for specific symbols
router.get('/quotes', async (req, res) => {
  try {
    const { symbols, start, end, limit } = req.query;

    const options = {
      start: start ? new Date(start) : undefined,
      end: end ? new Date(end) : undefined,
      limit: limit ? parseInt(limit) : 100
    };

    const symbolsArray = symbols.split(',');
    const quotes = alpaca.getCryptoQuotes(symbolsArray, options);
    const result = [];

    for await (let quote of quotes) {
      result.push(quote);
    }

    res.json(result);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to fetch crypto quotes' });
  }
});

// Get latest crypto quotes
router.get('/quotes/latest', async (req, res) => {
  try {
    const { symbols } = req.query;
    const symbolsArray = symbols.split(',');

    const quotes = await alpaca.getLatestCryptoQuotes(symbolsArray);
    res.json(Object.fromEntries(quotes));
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to fetch latest crypto quotes' });
  }
});

// Get crypto bars for specific symbols
router.get('/bars', async (req, res) => {
  try {
    const { symbols, start, end, timeframe, limit } = req.query;

    const options = {
      start: start ? new Date(start) : undefined,
      end: end ? new Date(end) : undefined,
      timeframe: timeframe || '1Min',  // Default to 1 minute bars
      limit: limit ? parseInt(limit) : 100
    };

    const symbolsArray = symbols.split(',');
    const bars = alpaca.getCryptoBars(symbolsArray, options);
    const result = [];

    for await (let bar of bars) {
      result.push(bar);
    }

    res.json(result);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to fetch crypto bars' });
  }
});

// Get latest crypto bars
router.get('/bars/latest', async (req, res) => {
  try {
    const { symbols } = req.query;
    const symbolsArray = symbols.split(',');

    const bars = await alpaca.getLatestCryptoBars(symbolsArray);
    res.json(Object.fromEntries(bars));
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to fetch latest crypto bars' });
  }
});

// Get crypto snapshots
router.get('/snapshots', async (req, res) => {
  try {
    const { symbols } = req.query;
    const symbolsArray = symbols.split(',');

    const snapshots = await alpaca.getCryptoSnapshots(symbolsArray);
    res.json(snapshots);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to fetch crypto snapshots' });
  }
});

export default router;
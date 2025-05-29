import express from 'express';
import { alpaca } from '../../client/index.js';  // Alpaca client configured here

const router = express.Router();

// Get all positions
router.get('/', async (req, res) => {
  try {
    const positions = await alpaca.getPositions();
    // console.log(positions);
    
    res.json(positions);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to fetch positions' });
  }
});

// Get a specific position by symbol
router.get('/:symbol', async (req, res) => {
  try {
    const { symbol } = req.params;
    const position = await alpaca.getPosition(symbol);
    res.json(position);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: `Failed to fetch position for symbol: ${symbol}` });
  }
});

// Close a specific position by symbol
router.delete('/:symbol', async (req, res) => {
  try {
    const { symbol } = req.params;
    await alpaca.closePosition(symbol);
    res.json({ message: `Position in ${symbol} closed successfully.` });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: `Failed to close position for symbol: ${symbol}` });
  }
});

// Close all positions
router.delete('/', async (req, res) => {
  try {
    await alpaca.closeAllPositions();
    res.json({ message: 'All positions closed successfully.' });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to close all positions' });
  }
});

export default router;
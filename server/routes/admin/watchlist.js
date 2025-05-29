import express from 'express';
import { alpaca } from '../../client/index.js';  // Alpaca client configured here

const router = express.Router();

// Get all watchlists
router.get('/', async (req, res) => {
  try {
    const watchlists = await alpaca.getWatchlists();
    res.json(watchlists);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to fetch watchlists' });
  }
});

// Get a specific watchlist by ID
router.get('/:id', async (req, res) => {
  try {
    const { id } = req.params;
    const watchlist = await alpaca.getWatchlist(id);
    res.json(watchlist);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: `Failed to fetch watchlist with ID: ${id}` });
  }
});

// Create a new watchlist
router.post('/', async (req, res) => {
  try {
    const { name, symbols } = req.body;
    const watchlist = await alpaca.addWatchlist(name, symbols || []);
    res.json(watchlist);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to create watchlist' });
  }
});

// Add a stock to an existing watchlist
router.post('/:id/add', async (req, res) => {
  try {
    const { id } = req.params;
    const { symbol } = req.body;
    const updatedWatchlist = await alpaca.addToWatchlist(id, symbol);
    res.json(updatedWatchlist);
  } catch (error) {
    console.error(error);
    // Check for duplicate symbol error
    if (error.response?.data?.code === 40010001) {
      res.status(422).json({ error: `Symbol ${req.body.symbol} is already in the watchlist` });
    } else {
      res.status(500).json({ error: `Failed to add symbol ${req.body.symbol} to watchlist ${req.params.id}` });
    }
  }
});

// Update a watchlist (replace stocks in the list)
router.put('/:id', async (req, res) => {
  try {
    const { id } = req.params;
    const { name, symbols } = req.body;
    const updatedWatchlist = await alpaca.updateWatchlist(id, symbols);
    res.json(updatedWatchlist);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: `Failed to update watchlist ${id}` });
  }
});

// Delete a watchlist by ID
router.delete('/:id', async (req, res) => {
  try {
    const { id } = req.params;
    await alpaca.deleteWatchlist(id);
    res.json({ message: `Watchlist with ID ${id} was deleted` });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: `Failed to delete watchlist with ID: ${id}` });
  }
});

// Delete a stock from a watchlist
router.delete('/:id/:symbol', async (req, res) => {
  try {
    const { id, symbol } = req.params;
    await alpaca.deleteFromWatchlist(id, symbol);
    res.json({ message: `Symbol ${symbol} was removed from watchlist ${id}` });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: `Failed to remove symbol ${symbol} from watchlist ${id}` });
  }
});

export default router;
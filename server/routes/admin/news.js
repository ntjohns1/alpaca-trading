import express from 'express';
import { alpaca } from '../../client/index.js';  // Alpaca client configured here

const router = express.Router();

// Get news based on query parameters
router.get('/', async (req, res) => {
  try {
    const { symbols, start, end, limit, sort } = req.query;

    // Construct options object for getNews
    const options = {
      symbols: symbols ? symbols.split(',') : undefined,  // Convert symbols to array
      start: start ? new Date(start) : undefined,  // Optional start date
      end: end ? new Date(end) : undefined,  // Optional end date
      limit: limit ? parseInt(limit) : 50,  // Default limit to 50
      sort: sort || 'desc'  // Default sort to descending
    };

    const news = await alpaca.getNews(options);
    res.json(news);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to fetch news' });
  }
});

export default router;
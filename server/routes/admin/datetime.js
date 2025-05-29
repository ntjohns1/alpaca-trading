import express from 'express';
import { alpaca } from '../../client/index.js';  // Alpaca client configured here

const router = express.Router();

// Get market calendar within a date range
router.get('/calendar', async (req, res) => {
  try {
    const { start, end } = req.query;

    // Call Alpaca's getCalendar method with optional start and end dates
    const calendar = await alpaca.getCalendar({
      start: start ? new Date(start) : undefined,
      end: end ? new Date(end) : undefined
    });

    res.json(calendar);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to fetch market calendar' });
  }
});

// Get market clock (is the market open or closed)
router.get('/clock', async (req, res) => {
  try {
    // Call Alpaca's getClock method to get current market clock status
    const clock = await alpaca.getClock();
    res.json(clock);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to fetch market clock' });
  }
});

export default router;
import express from 'express';
import { alpaca } from '../../client/index.js';

const router = express.Router();

// Get Account Info
router.get('/', async (req, res) => {
  try {
    const account = await alpaca.getAccount();
    res.json(account);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to fetch account details' });
  }
});

// Get Account Configurations
router.get('/configurations', async (req, res) => {
  try {
    const configurations = await alpaca.getAccountConfigurations();
    res.json(configurations);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to fetch account configurations' });
  }
});

// Update Account Configurations
router.patch('/configurations', async (req, res) => {
  try {
    const updatedConfig = await alpaca.updateAccountConfigurations(req.body);
    res.json(updatedConfig);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to update account configurations' });
  }
});

// Get Account Activities
router.get('/activities', async (req, res) => {
  try {
    const {
      activityTypes,
      until,
      after,
      direction,
      date,
      pageSize,
      pageToken
    } = req.query;

    const options = {
      activityTypes,
      until: until ? new Date(until) : undefined,
      after: after ? new Date(after) : undefined,
      direction,
      date: date ? new Date(date) : undefined,
      pageSize: pageSize ? parseInt(pageSize) : undefined,
      pageToken
    };

    const activities = await alpaca.getAccountActivities(options);
    res.json(activities);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to fetch account activities' });
  }
});

// Get Portfolio History
router.get('/portfolio/history', async (req, res) => {
  try {
    const {
      date_start,
      date_end,
      period,
      timeframe,
      extended_hours
    } = req.query;

    const options = {
      date_start,
      date_end,
      period,
      timeframe,
      extended_hours
    };

    const portfolioHistory = await alpaca.getPortfolioHistory(options);
    res.json(portfolioHistory);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to fetch portfolio history' });
  }
});

export default router;
import express from "express";
import { alpaca } from "../../client/index.js";

const router = express.Router();

// Get trades for a single symbol
router.get("/trades/:symbol", async (req, res) => {
  try {
    const { symbol } = req.params;
    const { start, end, limit } = req.query;

    const options = {
      start: start ? new Date(start) : undefined,
      end: end ? new Date(end) : undefined,
      limit: limit ? parseInt(limit) : undefined,
    };

    const trades = alpaca.getTradesV2(symbol, options);
    const result = [];

    for await (let trade of trades) {
      result.push(trade);
    }

    res.json(result);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: `Failed to fetch trades for ${symbol}` });
  }
});

// Get trades for multiple symbols
router.get("/multi-trades", async (req, res) => {
  try {
    const { symbols, start, end, limit } = req.query;

    const options = {
      start: start ? new Date(start) : undefined,
      end: end ? new Date(end) : undefined,
      limit: limit ? parseInt(limit) : undefined,
    };

    const symbolsArray = symbols.split(",");
    const trades = await alpaca.getMultiTradesV2(symbolsArray, options);
    res.json(Object.fromEntries(trades));
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Failed to fetch multiple symbol trades" });
  }
});

// Get latest trade for a single symbol
router.get("/trades/latest/:symbol", async (req, res) => {
  try {
    const { symbol } = req.params;
    const latestTrade = await alpaca.getLatestTrade(symbol);
    res.json(latestTrade);
  } catch (error) {
    console.error(error);
    res
      .status(500)
      .json({ error: `Failed to fetch latest trade for ${symbol}` });
  }
});

// Get latest trades for multiple symbols
router.get("/trades/latest", async (req, res) => {
  try {
    const { symbols } = req.query;
    const symbolsArray = symbols.split(",");
    const latestTrades = await alpaca.getLatestTrades(symbolsArray);
    res.json(Object.fromEntries(latestTrades));
  } catch (error) {
    console.error(error);
    res
      .status(500)
      .json({ error: "Failed to fetch latest trades for multiple symbols" });
  }
});

// Get quotes for a single symbol
router.get("/quotes/:symbol", async (req, res) => {
  try {
    const { symbol } = req.params;
    const { start, end, limit } = req.query;

    const options = {
      start: start ? new Date(start) : undefined,
      end: end ? new Date(end) : undefined,
      limit: limit ? parseInt(limit) : undefined,
    };

    const quotes = alpaca.getQuotesV2(symbol, options);
    const result = [];

    for await (let quote of quotes) {
      result.push(quote);
    }

    res.json(result);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: `Failed to fetch quotes for ${symbol}` });
  }
});

// Get latest quote for a single symbol
router.get("/quotes/latest/:symbol", async (req, res) => {
  try {
    const { symbol } = req.params;
    const latestQuote = await alpaca.getLatestQuote(symbol);
    res.json(latestQuote);
  } catch (error) {
    console.error(error);
    res
      .status(500)
      .json({ error: `Failed to fetch latest quote for ${symbol}` });
  }
});

// Get latest quotes for multiple symbols
router.get("/quotes/latest", async (req, res) => {
  try {
    const { symbols } = req.query;
    const symbolsArray = symbols.split(",");
    const latestQuotes = await alpaca.getLatestQuotes(symbolsArray);
    res.json(Object.fromEntries(latestQuotes));
  } catch (error) {
    console.error(error);
    res
      .status(500)
      .json({ error: "Failed to fetch latest quotes for multiple symbols" });
  }
});

const getFormattedDate = (date) => date.toISOString().slice(0, 10);

const today = new Date();
const yesterday = new Date(today);
yesterday.setDate(yesterday.getDate() - 1);

router.get("/:symbol", async (req, res) => {
  try {
    const { symbol } = req.params;
    const { start, end, timeframe, timeframeUnit, limit } = req.query;
    // console.log(req.query);

    const options = {
      start: start || getFormattedDate(yesterday),
      end: end || getFormattedDate(today),
      timeframe: alpaca.newTimeframe(
        parseInt(timeframe),
        alpaca.timeframeUnit[timeframeUnit]
      ),
      limit: parseInt(limit) || 1000,
    };

    const bars = alpaca.getBarsV2(symbol, options);
    const result = [];
    for await (let bar of bars) {
      result.push(bar);
    }

    res.json(result);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Failed to fetch bars" });
  }
});

router.get("/multibars", async (req, res) => {
  try {
    const { start, end, timeframe, timeframeUnit, limit } = req.query;
    const symbolsArray = symbols.split(",");

    const options = {
      start: start || getFormattedDate(yesterday),
      end: end || getFormattedDate(today),
      timeframe: alpaca.newTimeframe(
        parseInt(timeframe),
        alpaca.timeframeUnit[timeframeUnit]
      ),
      limit: parseInt(limit) || 2,
    };

    const bars = await alpaca.getMultiBarsV2(symbolsArray, options);
    res.json(Object.fromEntries(bars));
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Failed to fetch multi-symbol bars" });
  }
});

router.get("/latestbar/:symbol", async (req, res) => {
  try {
    const { symbol } = req.params;
    const latestBar = await alpaca.getLatestBar(symbol);
    res.json(latestBar);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Failed to fetch latest bar" });
  }
});

router.get("/latestbars", async (req, res) => {
  try {
    const { symbols } = req.query;
    const symbolsArray = symbols.split(",");

    const latestBars = await alpaca.getLatestBars(symbolsArray);
    res.json(Object.fromEntries(latestBars));
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Failed to fetch latest bars" });
  }
});

router.get("/multibars-async", async (req, res) => {
  try {
    const { symbols, start, end, timeframeUnit, limit } = req.query;
    const symbolsArray = symbols.split(",");

    const options = {
      start: start || getFormattedDate(yesterday),
      end: end || getFormattedDate(today),
      timeframe: alpaca.alpaca.newTimeframe(
        30,
        alpaca.timeframeUnit[timeframeUnit.toUpperCase()]
      ),
      limit: parseInt(limit) || 2,
    };

    const barsAsync = alpaca.getMultiBarsAsyncV2(symbolsArray, options);
    const result = [];

    for await (let bar of barsAsync) {
      result.push(bar);
    }

    res.json(result);
  } catch (error) {
    console.error(error);
    res
      .status(500)
      .json({ error: "Failed to fetch multi-symbol bars asynchronously" });
  }
});

// Get snapshot for a single symbol
router.get("/snapshot/:symbol", async (req, res) => {
  try {
    const { symbol } = req.params;
    const snapshot = await alpaca.getSnapshot(symbol);
    res.json(snapshot);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: `Failed to fetch snapshot for ${symbol}` });
  }
});

// Get snapshots for multiple symbols
router.get("/snapshots", async (req, res) => {
  try {
    const { symbols } = req.query;
    const symbolsArray = symbols.split(",");
    const snapshots = await alpaca.getSnapshots(symbolsArray);
    res.json(snapshots);
  } catch (error) {
    console.error(error);
    res
      .status(500)
      .json({ error: "Failed to fetch snapshots for multiple symbols" });
  }
});

export default router;

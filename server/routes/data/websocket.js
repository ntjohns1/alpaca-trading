import express from "express";
import { WebSocketServer } from "ws";
import { alpaca } from "../../client/index.js";

const router = express.Router();
const wss = new WebSocketServer({ noServer: true });
const clients = new Set();
const clientSubscriptions = new Map(); // Maps clients to their subscribed symbols

const socket = alpaca.data_stream_v2;

// Handle WebSocket connections
wss.on("connection", (ws) => {
  console.log("New WebSocket client connected");
  clients.add(ws);
  clientSubscriptions.set(ws, new Set());

  ws.on("message", (data) => {
    try {
      const message = JSON.parse(data);
      console.log("Received message from WebSocket client:", message);

      if (message.action === "subscribe") {
        const symbols = new Set([
          ...(message.quotes || []),
          ...(message.trades || []),
          ...(message.bars || [])
        ]);

        clientSubscriptions.set(ws, symbols);
        
        // Subscribe to Alpaca streams
        if (message.quotes?.length) {
          console.log("🟢 Subscribing to quotes for:", message.quotes);
          socket.subscribeForQuotes(message.quotes);
        }
        if (message.trades?.length) {
          console.log("🟢 Subscribing to trades for:", message.trades);
          socket.subscribeForTrades(message.trades);
        }
        if (message.bars?.length) {
          console.log("🟢 Subscribing to bars for:", message.bars);
          socket.subscribeForBars(message.bars);
        }
      } else if (message.action === "unsubscribe") {
        const symbols = new Set([
          ...(message.quotes || []),
          ...(message.trades || []),
          ...(message.bars || [])
        ]);

        // Update client subscriptions
        const currentSubs = clientSubscriptions.get(ws);
        symbols.forEach(symbol => currentSubs.delete(symbol));

        // Unsubscribe from Alpaca streams if no other clients need them
        if (message.quotes?.length) {
          console.log("🔴 Unsubscribing from quotes for:", message.quotes);
          socket.unsubscribeFromQuotes(message.quotes);
        }
        if (message.trades?.length) {
          console.log("🔴 Unsubscribing from trades for:", message.trades);
          socket.unsubscribeFromTrades(message.trades);
        }
        if (message.bars?.length) {
          console.log("🔴 Unsubscribing from bars for:", message.bars);
          socket.unsubscribeFromBars(message.bars);
        }
      }
    } catch (err) {
      console.error("❌ Invalid message from WebSocket client:", err);
    }
  });

  ws.on("close", () => {
    console.log("WebSocket client disconnected");
    clients.delete(ws);
    clientSubscriptions.delete(ws);
  });
});

// Handle Alpaca WebSocket events
socket.onConnect(() => {
  console.log("Connected to Alpaca data stream");
});

socket.onStockTrade((trade) => {
  // console.log("Trade:", trade);
  broadcast({ type: "trade", data: trade });
});

socket.onStockQuote((quote) => {
  // console.log("Quote:", quote);
  broadcast({ type: "quote", data: quote });
});

socket.onStockBar((bar) => {
  // console.log("Bar:", bar);
  broadcast({ type: "bar", data: bar });
});

socket.onDisconnect(() => {
  console.log("Disconnected from Alpaca data stream");
});

socket.connect();

// Filtered Broadcast function (based on symbols)
const broadcast = (data) => {
  for (const client of clients) {
    if (client.readyState === 1) {
      const symbols = clientSubscriptions.get(client) || new Set();
      // Handle both quote and bar formats
      const symbol = data.type === 'quote' ? data.data.Symbol : data.data.S;
      
      if (symbols.has(symbol)) {
        try {
          client.send(JSON.stringify(data));
        } catch (error) {
          console.error('Error sending data to client:', error);
        }
      }
    }
  }
};

// WebSocket Upgrade Handling
router.upgrade = (request, socket, head) => {
  console.log("Handling WebSocket upgrade request...");
  wss.handleUpgrade(request, socket, head, (ws) => {
    console.log("WebSocket upgrade successful.");
    wss.emit("connection", ws, request);
  });
};

export default router;
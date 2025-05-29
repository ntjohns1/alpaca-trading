import "dotenv/config";
import express from "express";
import http from "http";
import cors from "cors";
import { promisify } from "util";
import { connect } from "./config/index.js";
import { authMiddleware } from "./middleware/index.js";
import {
  accounts,
  assets,
  datetime,
  news,
  orders,
  positions,
  watchlist,
  stocks,
  crypto,
  options,
  websocket,
} from "./routes/index.js";
const app = express();
const server = http.createServer(app);

const corsOptions = {
  origin: "http://localhost:3000",
  methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
  credentials: true,
};
app.use(cors(corsOptions));
app.use(authMiddleware);
app.use(express.json());

// Alpaca API routes
app.use("/api/stocks", stocks);
app.use("/api/crypto", crypto);
app.use("/api/options", options);
app.use("/api/accounts", accounts);
app.use("/api/assets", assets);
app.use("/api/datetime", datetime);
app.use("/api/news", news);
app.use("/api/orders", orders);
app.use("/api/positions", positions);
app.use("/api/watchlist", watchlist);
app.use("/api/stream", websocket);

const startServer = async () => {
  await connect();
  const port = process.env.SERVER_PORT || 8080;
  await promisify(server.listen).bind(server)(port);
  console.log(`Listening on port ${port}`);
};

// WebSocket Upgrade Handling
server.on("upgrade", (request, socket, head) => {
  console.log("Received WebSocket upgrade request:", request.url);
  websocket.upgrade(request, socket, head);
});

startServer();

import express from 'express';
import { alpaca } from '../../client/index.js';  // Alpaca client configured here

const router = express.Router();

// Create a new order
router.post('', async (req, res) => {
  try {
    const {
      symbol,
      qty,
      notional,
      side,
      type,
      time_in_force,
      limit_price,
      stop_price,
      client_order_id,
      extended_hours,
      order_class,
      take_profit,
      stop_loss,
      trail_price,
      trail_percent
    } = req.body;

    // Alpaca's createOrder function
    const order = await alpaca.createOrder({
      symbol,
      qty,
      notional,
      side,
      type,
      time_in_force,
      limit_price,
      stop_price,
      client_order_id,
      extended_hours,
      order_class,
      take_profit,
      stop_loss,
      trail_price,
      trail_percent
    });

    res.json(order);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to create order' });
  }
});

// Get list of orders
router.get('/', async (req, res) => {
  try {
    const { status, after, until, limit, direction } = req.query;

    const options = {
      status: status || 'open',
      after: after ? new Date(after) : undefined,
      until: until ? new Date(until) : undefined,
      limit: limit ? parseInt(limit) : undefined,
      direction
    };

    const orders = await alpaca.getOrders(options);
    res.json(orders);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to fetch orders' });
  }
});

// Get order by ID
router.get('/:id', async (req, res) => {
  try {
    const { id } = req.params;
    const order = await alpaca.getOrder(id);
    res.json(order);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: `Failed to fetch order with ID: ${id}` });
  }
});

// Get order by client_order_id
router.get('/by_client_order_id/:client_order_id', async (req, res) => {
  try {
    const { client_order_id } = req.params;
    const order = await alpaca.getOrderByClientId(client_order_id);
    res.json(order);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: `Failed to fetch order with Client Order ID: ${client_order_id}` });
  }
});

// Update an existing order by ID
router.patch('/:id', async (req, res) => {
  try {
    const { id } = req.params;
    const updatedOrder = await alpaca.replaceOrder(id, req.body);
    res.json(updatedOrder);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: `Failed to update order with ID: ${id}` });
  }
});

// Cancel an order by ID
router.delete('/:id', async (req, res) => {
  try {
    const { id } = req.params;
    await alpaca.cancelOrder(id);
    res.json({ message: `Order with ID: ${id} was canceled` });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: `Failed to cancel order with ID: ${id}` });
  }
});

// Cancel all open orders
router.delete('/', async (req, res) => {
  try {
    await alpaca.cancelAllOrders();
    res.json({ message: 'All open orders were canceled' });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to cancel all orders' });
  }
});

export default router;
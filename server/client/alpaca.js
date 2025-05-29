import Alpaca from "@alpacahq/alpaca-trade-api";
const API_URL = process.env.ALPACA_API_URL;
const API_KEY = process.env.ALPACA_API_KEY;
const API_SECRET = process.env.ALPACA_API_SECRET;

const alpaca = new Alpaca({
  keyId: API_KEY,
  secretKey: API_SECRET,
  paper: true,
  
});
// let options = {
//   start: "2022-09-01",
//   end: "2022-09-07",
//   timeframe: alpaca.newTimeframe(1, alpaca.timeframeUnit.DAY),
// };
// const testCall = async () => {
//   const bars = await alpaca.getCryptoBars(["BTC/USD"], options);

//   console.table(bars.get("BTC/USD"));
// };

//DataV2
// getTradesV2(symbol, options, config = this.configuration) {
//   return dataV2.getTrades(symbol, options, config);
// }
// getMultiTradesV2(symbols, options, config = this.configuration) {
//   return dataV2.getMultiTrades(symbols, options, config);
// }
// getMultiTradesAsyncV2(symbols, options, config = this.configuration) {
//   return dataV2.getMultiTradesAsync(symbols, options, config);
// }
// getQuotesV2(symbol, options, config = this.configuration) {
//   return dataV2.getQuotes(symbol, options, config);
// }
// getMultiQuotesV2(symbols, options, config = this.configuration) {
//   return dataV2.getMultiQuotes(symbols, options, config);
// }
// getMultiQuotesAsyncV2(symbols, options, config = this.configuration) {
//   return dataV2.getMultiQuotesAsync(symbols, options, config);
// }
// getBarsV2(symbol, options, config = this.configuration) {
//   return dataV2.getBars(symbol, options, config);
// }
// getMultiBarsV2(symbols, options, config = this.configuration) {
//   return dataV2.getMultiBars(symbols, options, config);
// }
// getMultiBarsAsyncV2(symbols, options, config = this.configuration) {
//   return dataV2.getMultiBarsAsync(symbols, options, config);
// }
// getLatestTrade(symbol, config = this.configuration) {
//   return dataV2.getLatestTrade(symbol, config);
// }
// getLatestTrades(symbols, config = this.configuration) {
//   return dataV2.getLatestTrades(symbols, config);
// }
// getLatestQuote(symbol, config = this.configuration) {
//   return dataV2.getLatestQuote(symbol, config);
// }
// getLatestQuotes(symbols, config = this.configuration) {
//   return dataV2.getLatestQuotes(symbols, config);
// }
// getLatestBar(symbol, config = this.configuration) {
//   return dataV2.getLatestBar(symbol, config);
// }
// getLatestBars(symbols, config = this.configuration) {
//   return dataV2.getLatestBars(symbols, config);
// }
// getSnapshot(symbol, config = this.configuration) {
//   return dataV2.getSnapshot(symbol, config);
// }
// getSnapshots(symbols, config = this.configuration) {
//   return dataV2.getSnapshots(symbols, config);
// }
export default alpaca;
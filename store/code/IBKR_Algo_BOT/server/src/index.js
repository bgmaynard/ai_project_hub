const express = require("express");
const cors = require("cors");
const path = require("path");
require("dotenv").config();

// Ports (env wins)
const PORT    = Number(process.env.SERVER_PORT || 3000);
const WS_PORT = Number(process.env.WS_PORT     || 8080);

const app = express();
app.use(cors());
app.use(express.json());

// === UI directory resolution ===
// 1) If UI_DIR env provided, use it.
// 2) Otherwise, go up TWO levels from server\src -> ...\IBKR_Algo_BOT\ui
const UI_DIR = process.env.UI_DIR
  ? path.resolve(process.env.UI_DIR)
  : path.resolve(__dirname, "..", "..", "ui");

app.use(express.static(UI_DIR));

app.get("/api/status", (req,res)=>{
  res.json({ server:"running", uiDir:UI_DIR, ts:new Date().toISOString() });
});

// Fallback to index.html
app.get("*", (req,res)=> res.sendFile(path.join(UI_DIR, "index.html")));

const server = app.listen(PORT, ()=>{
  console.log(`Co-located server online: http://localhost:${PORT}`);
  console.log(`Serving UI from: ${UI_DIR}`);
});

try {
  const WebSocket = require("ws");
  const wss = new WebSocket.Server({ port: WS_PORT });
  wss.on("connection", ws => {
    ws.send(JSON.stringify({ type:"connected", ts:new Date().toISOString() }));
  });
  console.log(`WS listening on ws://localhost:${WS_PORT}`);
} catch(e) {
  console.log("WS disabled:", e.message);
}

# Claude / Copilot Review Instructions  
*(AI Collaboration Brief for IBKR Algo Bot Project)*

## 🧠 Objective
You are assisting in reviewing and improving the **IBKR Algo Bot** codebase:
> https://github.com/bgmaynard/ai_project_hub

The project is a **FastAPI** service that bridges Interactive Brokers TWS/IB Gateway using **ib_insync**, enabling algo trade execution, order previewing, and AI strategy hooks.

---

## 🔍 Current Issue (to diagnose)
Service starts but IBKR connection fails with:
\\\
ConnectionRefusedError(22, 'The remote computer refused the network connection', None, 1225, None)
\\\

TWS API settings (confirmed):
- Host: 127.0.0.1
- Socket port: **7497**
- Master API Client ID: **6001**
- Allow connections from localhost only: **ON**
- Enable ActiveX and Socket Clients: **ON**

Suspects:
- Host/port/clientId handling in code
- Retry/timeouts/async usage around ib.connectAsync
- Env parsing (string -> int)
- Startup watchdog and error handling clarity

---

## 🧩 Files to Review First
| Path | Purpose |
|------|---------|
| \store/code/IBKR_Algo_BOT/bridge/ib_adapter.py\ | ib_insync connection + config |
| \store/code/IBKR_Algo_BOT/dashboard_api.py\ | FastAPI app, /api/status, /api/tws/ping, /api/order/preview |
| \scripts/bootstrap.ps1\ | Windows bootstrap + uvicorn launcher |
| \.env.example\ | Reference env vars for TWS host/port/clientId |

---

## 💡 Review Goals
1. Find the *root cause* of the refused connection and propose the **minimal** safe fix.
2. Ensure robust retries + timeouts without blocking the event loop.
3. Harden env parsing (e.g., ints for port/clientId).
4. Improve diagnostics (clear logs for host/port/clientId, isConnected() checks).
5. Keep Windows 11 + PowerShell flow smooth.

---

## 🧭 Runtime Context
- Python 3.11
- Key deps: fastapi, uvicorn, ib-insync, pydantic, python-dotenv
- Entrypoint:
  \\\powershell
  .\.venv\Scripts\python -m uvicorn store.code.IBKR_Algo_BOT.dashboard_api:app --host 127.0.0.1 --port 9101
  \\\

---

## ⚙️ Expected Output from Review
- Short summary of detected issues (with file/line refs if possible)
- Specific code diffs/snippets to fix connection issues
- Confirmation criteria: /api/tws/ping reports connected = true

---

## 📌 Quick Prompts to Paste in Claude / Copilot
**Claude prompt:**
> Review \store/code/IBKR_Algo_BOT/bridge/ib_adapter.py\ and \store/code/IBKR_Algo_BOT/dashboard_api.py\.  
> Diagnose why IBKR connection fails with ConnectionRefusedError on 127.0.0.1:7497, clientId 6001.  
> Recommend minimal code changes to make \ib.connectAsync\ succeed, with robust timeout/retry/logging and proper env parsing.  
> Return concrete diffs/snippets.

**Copilot prompt:**
> Propose a small patch to ib_adapter.py to:
> - log host/port/clientId before connect
> - use \	imeout=...\ on \connectAsync\
> - convert env vars to correct types with defaults
> - retry a few clientIds if 326/connection refused  
> Then adjust dashboard_api.py status/ping handlers to reflect \ib.isConnected()\.

---

## 🤝 Workflow
1. Claude suggests targeted fixes → commit to \eat/ibkr-adapter-collab-brief\.
2. Copilot refines or validates via inline suggestions.
3. Test locally:  
   - \GET /api/tws/ping\ → expect connected: true  
   - \GET /api/status\ → host=127.0.0.1, port=7497, clientId=6001

**Maintainer:** Bob Maynard  
**Created:** 2025-10-19 · **Revision:** 1.0

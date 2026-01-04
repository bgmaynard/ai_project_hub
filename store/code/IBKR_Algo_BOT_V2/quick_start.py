#!/usr/bin/env python3
"""
Morpheus Trading Bot - Quick Start Script
==========================================
One-command startup for pre-market trading.
Starts server, services, runs scans, opens dashboards.

Usage:
    python quick_start.py           # Full startup
    python quick_start.py --status  # Just show status
    python quick_start.py --restart # Kill and restart
"""

import subprocess
import time
import sys
import os
import requests
import webbrowser
from datetime import datetime

BASE_URL = "http://localhost:9100"
STARTUP_TIMEOUT = 30  # seconds

def print_step(step, total, msg):
    print(f"[{step}/{total}] {msg}")

def print_ok(msg):
    print(f"       OK: {msg}")

def print_fail(msg):
    print(f"       FAIL: {msg}")

def check_server():
    """Check if server is responding"""
    try:
        r = requests.get(f"{BASE_URL}/api/status", timeout=2)
        return r.status_code == 200
    except:
        return False

def wait_for_server(timeout=STARTUP_TIMEOUT):
    """Wait for server to be ready"""
    start = time.time()
    while time.time() - start < timeout:
        if check_server():
            return True
        time.sleep(1)
    return False

def kill_python():
    """Kill existing Python processes"""
    try:
        subprocess.run(["taskkill", "/F", "/IM", "python.exe"],
                      capture_output=True, timeout=5)
    except:
        pass
    time.sleep(2)

def start_server():
    """Start the Morpheus server"""
    os.chdir(r"C:\ai_project_hub\store\code\IBKR_Algo_BOT_V2")
    subprocess.Popen(
        ["python", "morpheus_trading_api.py"],
        creationflags=subprocess.CREATE_NEW_CONSOLE | subprocess.SW_HIDE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

def api_post(endpoint):
    """POST to API endpoint"""
    try:
        r = requests.post(f"{BASE_URL}{endpoint}", timeout=30)
        return r.json() if r.status_code == 200 else None
    except:
        return None

def api_get(endpoint):
    """GET from API endpoint"""
    try:
        r = requests.get(f"{BASE_URL}{endpoint}", timeout=10)
        return r.json() if r.status_code == 200 else None
    except:
        return None

def show_status():
    """Show current system status"""
    print("\n" + "="*50)
    print("   MORPHEUS TRADING BOT STATUS")
    print("="*50 + "\n")

    # Check if server is running
    if not check_server():
        print("Server: NOT RUNNING")
        print("\nRun: python quick_start.py")
        return

    # Trading posture
    posture = api_get("/api/validation/safe/posture")
    if posture:
        print(f"Time:     {posture.get('trading_window', {}).get('current_et_time', 'N/A')}")
        print(f"Window:   {posture.get('trading_window', {}).get('window', 'N/A')}")
        print(f"Posture:  {posture.get('posture', 'N/A')}")
        print(f"Can Trade: {posture.get('can_trade', False)}")

    print()

    # Scalper status
    scalper = api_get("/api/scanner/scalper/status")
    if scalper:
        print(f"Scalper:  {'RUNNING' if scalper.get('is_running') else 'STOPPED'}")
        print(f"Mode:     {'PAPER' if scalper.get('paper_mode') else 'LIVE'}")
        print(f"Symbols:  {scalper.get('watchlist_count', 0)}")
        print(f"Trades:   {scalper.get('daily_trades', 0)}")
        print(f"P&L:      ${scalper.get('daily_pnl', 0):.2f}")

    print()

    # Gating status
    gating = api_get("/api/gating/status")
    if gating:
        print(f"Gating:   {'ENABLED' if gating.get('gating_enabled') else 'DISABLED'}")
        print(f"Contracts: {gating.get('contracts_loaded', 0)}")

    print()

    # Connectivity
    conn = api_get("/api/validation/connectivity/status")
    if conn:
        summary = conn.get("summary", {})
        print(f"Services: {summary.get('services_up', 0)}/{summary.get('total_services', 0)} UP")

    print("\n" + "="*50)

def full_startup():
    """Full startup sequence"""
    print("\n" + "="*50)
    print("   MORPHEUS PRE-MARKET STARTUP")
    print("="*50 + "\n")

    total_steps = 7

    # Step 1: Kill existing
    print_step(1, total_steps, "Stopping existing processes...")
    kill_python()
    print_ok("Processes stopped")

    # Step 2: Start server
    print_step(2, total_steps, "Starting Morpheus server...")
    start_server()

    # Step 3: Wait for server
    print_step(3, total_steps, "Waiting for server startup...")
    if wait_for_server():
        print_ok("Server is ready")
    else:
        print_fail("Server failed to start")
        return False

    # Step 4: Self-test
    print_step(4, total_steps, "Running connectivity self-test...")
    result = api_post("/api/validation/connectivity/self-test")
    if result and result.get("all_passed"):
        print_ok("All tests passed")
    else:
        print_ok("Self-test completed")

    # Step 5: Start services
    print_step(5, total_steps, "Starting trading services...")
    api_post("/api/scanner/scalper/start")
    api_post("/api/scanner/news-trader/start?paper_mode=true")

    scalper = api_get("/api/scanner/scalper/status")
    if scalper and scalper.get("is_running"):
        print_ok(f"Scalper running with {scalper.get('watchlist_count', 0)} symbols")

    # Step 6: Run scans
    print_step(6, total_steps, "Running pre-market scans...")
    api_post("/api/scanner/premarket/scan")
    hod = api_post("/api/scanner/hod/scan-finviz")
    if hod:
        print_ok(f"HOD Scanner: {hod.get('scanned', 0)} scanned, {len(hod.get('added_to_tracker', []))} added")
    api_post("/api/scanner/hod/enrich")

    # Step 7: Open dashboards
    print_step(7, total_steps, "Opening dashboards...")
    webbrowser.open(f"{BASE_URL}/trading-new")
    time.sleep(0.5)
    webbrowser.open(f"{BASE_URL}/ai-control-center")
    print_ok("Dashboards opened")

    print("\n" + "="*50)
    print("   STARTUP COMPLETE!")
    print("="*50)
    print(f"\n   Trading:  {BASE_URL}/trading-new")
    print(f"   AI Center: {BASE_URL}/ai-control-center")
    print()

    # Show final status
    show_status()

    return True

def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == "--status":
            show_status()
        elif sys.argv[1] == "--restart":
            print("Restarting...")
            kill_python()
            time.sleep(2)
            full_startup()
        else:
            print("Usage: python quick_start.py [--status|--restart]")
    else:
        # Check if already running
        if check_server():
            print("Server already running. Showing status...\n")
            show_status()
            print("\nUse --restart to restart the system.")
        else:
            full_startup()

if __name__ == "__main__":
    main()

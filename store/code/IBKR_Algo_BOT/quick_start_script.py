"""
Quick Start Script - Creates all dashboard files
Run this to generate the complete dashboard system
"""

import os
from pathlib import Path

print("=" * 60)
print("  Creating Trading Dashboard Files")
print("=" * 60)

# Full dashboard_api.py content
dashboard_api_content = '''"""
Trading Bot Dashboard API Server
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import threading
import time
import logging
from datetime import datetime

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
class BotState:
    def __init__(self):
        self.mtf_bot = {
            'status': 'stopped',
            'active_positions': [],
            'recent_trades': [],
            'pnl': 0.0
        }
        self.warrior_bot = {
            'status': 'stopped',
            'active_positions': [],
            'recent_trades': [],
            'pnl': 0.0,
            'gappers_found': []
        }
        self.ibkr = {
            'connected': False,
            'port': 7497,
            'account_value': 1000000.0
        }
        self.activity_log = []
    
    def add_log(self, level, source, message):
        entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'source': source,
            'message': message
        }
        self.activity_log.insert(0, entry)
        if len(self.activity_log) > 100:
            self.activity_log = self.activity_log[:100]
        socketio.emit('log_update', entry)
        return entry

bot_state = BotState()

# API Endpoints
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/api/status', methods=['GET'])
def get_system_status():
    return jsonify({
        'mtf': bot_state.mtf_bot,
        'warrior': bot_state.warrior_bot,
        'ibkr': bot_state.ibkr,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/mtf/start', methods=['POST'])
def start_mtf_bot():
    bot_state.mtf_bot['status'] = 'running'
    bot_state.add_log('success', 'mtf', 'MTF bot started')
    return jsonify({'success': True, 'message': 'MTF bot started'})

@app.route('/api/mtf/stop', methods=['POST'])
def stop_mtf_bot():
    bot_state.mtf_bot['status'] = 'stopped'
    bot_state.add_log('info', 'mtf', 'MTF bot stopped')
    return jsonify({'success': True, 'message': 'MTF bot stopped'})

@app.route('/api/warrior/start', methods=['POST'])
def start_warrior_scanner():
    bot_state.warrior_bot['status'] = 'running'
    bot_state.add_log('success', 'warrior', 'Warrior scanner started')
    return jsonify({'success': True, 'message': 'Warrior scanner started'})

@app.route('/api/warrior/stop', methods=['POST'])
def stop_warrior_scanner():
    bot_state.warrior_bot['status'] = 'stopped'
    bot_state.add_log('info', 'warrior', 'Warrior scanner stopped')
    return jsonify({'success': True, 'message': 'Warrior scanner stopped'})

@app.route('/api/positions', methods=['GET'])
def get_positions():
    all_positions = bot_state.mtf_bot['active_positions'] + bot_state.warrior_bot['active_positions']
    return jsonify({'positions': all_positions, 'total_count': len(all_positions)})

@app.route('/api/trades', methods=['GET'])
def get_trades():
    all_trades = bot_state.mtf_bot['recent_trades'] + bot_state.warrior_bot['recent_trades']
    all_trades.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    return jsonify({'trades': all_trades[:50]})

@app.route('/api/logs', methods=['GET'])
def get_activity_logs():
    return jsonify({'logs': bot_state.activity_log[:100]})

@app.route('/api/pnl', methods=['GET'])
def get_pnl_summary():
    return jsonify({
        'mtf': {'current': bot_state.mtf_bot['pnl']},
        'warrior': {'current': bot_state.warrior_bot['pnl']},
        'total': bot_state.mtf_bot['pnl'] + bot_state.warrior_bot['pnl']
    })

@app.route('/api/emergency-stop', methods=['POST'])
def emergency_stop():
    bot_state.add_log('warning', 'system', 'EMERGENCY STOP INITIATED')
    bot_state.mtf_bot['status'] = 'stopped'
    bot_state.warrior_bot['status'] = 'stopped'
    return jsonify({'success': True, 'message': 'All systems stopped'})

# WebSocket handlers
@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')
    emit('connection_status', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected')

# Background status monitor
def status_monitor():
    while True:
        time.sleep(5)
        socketio.emit('status_update', {
            'mtf': bot_state.mtf_bot,
            'warrior': bot_state.warrior_bot,
            'ibkr': bot_state.ibkr
        })

def start_background_tasks():
    monitor_thread = threading.Thread(target=status_monitor, daemon=True)
    monitor_thread.start()

if __name__ == '__main__':
    logger.info('Starting Trading Bot Dashboard API Server...')
    bot_state.add_log('info', 'system', 'Dashboard API server starting...')
    start_background_tasks()
    logger.info('Server running on http://localhost:5000')
    bot_state.add_log('success', 'system', 'Dashboard API server ready')
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)
'''

# Write the file
print("\n[1/3] Creating dashboard_api.py...")
with open('dashboard_api.py', 'w') as f:
    f.write(dashboard_api_content)
print("      ✓ Created dashboard_api.py")

# Create start script
print("\n[2/3] Creating start_backend.bat...")
start_script = '''@echo off
echo ========================================
echo   Trading Dashboard Backend
echo ========================================
echo.
echo Starting API server on http://localhost:5000
echo.
echo Press Ctrl+C to stop
echo.
python dashboard_api.py
pause
'''

with open('start_backend.bat', 'w') as f:
    f.write(start_script)
print("      ✓ Created start_backend.bat")

# Create directories
print("\n[3/3] Creating directories...")
os.makedirs('dashboard_data', exist_ok=True)
os.makedirs('frontend/src', exist_ok=True)
print("      ✓ Created directories")

print("\n" + "=" * 60)
print("  ✅ Setup Complete!")
print("=" * 60)
print("\nYour dashboard files are ready!")
print("\nTo start the backend:")
print("  Option 1: .\\start_backend.bat")
print("  Option 2: python dashboard_api.py")
print("\nAPI will be available at: http://localhost:5000")
print("\nTest it:")
print("  Browser: http://localhost:5000/api/health")
print("  PowerShell: (Invoke-WebRequest http://localhost:5000/api/health).Content")
print("\n" + "=" * 60)


    IBKR ALGO TRADING BOT - QUICK START
    ====================================
    
    1. Start the Dashboard:
       python run_dashboard.py
       
    2. Access the UI:
       http://127.0.0.1:9101/ui/
       
    3. Run AI Validation:
       python ai_mesh_controller.py
       
    4. Push Changes:
       ./scripts/push_patch.ps1 -Message "cont: your message"
       
    5. Check Status:
       - Mesh Heartbeat: store/ai_shared/mesh_heartbeat.json
       - Validator Report: store/ai_shared/validator_report.md
       - Trainer Audit: store/ai_shared/trainer_audit.md
    
    API Endpoints:
    - Account Info: GET /api/account
    - Positions: GET /api/positions
    - Orders: GET/POST /api/orders
    - Strategies: GET/PUT /api/strategies
    - Trade Signals: GET /api/signals
    - WebSocket: ws://127.0.0.1:9101/ws
    
    For more info, see AI_PROJECT_HUB_COLLABORATION_GUIDE.md
    
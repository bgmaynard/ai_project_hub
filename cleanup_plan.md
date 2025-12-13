# Repository Cleanup Plan

## Files and Directories to Remove

### 1. Root Directory - Backup & Old Files (15 files)
- `dashboard_api_fixed.py` - old fixed version
- `dashboard_api_hold.py` - temporary hold file
- `dashboard_api_immediate.py` - old version
- `ib_adapter.py` - duplicate/old
- `ibkr_adapter_fixed.py` - old fixed version
- `ai_router_complete_package.zip` - archived package
- `files.zip` - archived files
- `Run-CollabCycle.ps1.bak_*` (4 files) - backup scripts

### 2. Root Directory - Test Files (7 files)
- `test_adapter.py`
- `test_adapter_standalone.py`
- `test_forced_id.py`
- `test_ids.py`
- `test_long.py`
- `test_tws.py`
- `test_tws_7497.py`

### 3. Root Directory - Old Install Scripts (3 files)
- `INSTALL_COLOCATED_SERVER.bat` - outdated
- `INSTALL_GUI_ONLY.bat` - outdated
- `INSTALL_SERVER.bat` - outdated

### 4. Root Directory - Self-Install Scripts (3 files)
- `SELF_INSTALL.bat` - outdated
- `SELF_INSTALL_COMBO.bat` - outdated
- `SELF_INSTALL_REAL_IBKR.bat` - outdated

### 5. Root Directory - Old HTML Files (3 files)
- `status.html` - old dashboard
- `trading.html` - old UI
- `trading_hub.html` - old UI

### 6. Root Directory - Redundant Scripts
- `apply_ibkr_connection_patch.py` - empty file
- `self_installing_patch.py` - old patch
- `setup_ibkr_bot.py` - superseded
- `setup_everything.py` - superseded
- `ai_mesh_controller.py` - unused

### 7. Backups Directory - Entire deployment backup
- `backups/deployment_backup_20251112_202939/` - HUGE recursive backup (problematic)

### 8. IBKR_Algo_BOT - Backup Files (20+ .bak files)
- `store/code/IBKR_Algo_BOT/dashboard_api.py.*.bak` - multiple dated backups
- `store/code/IBKR_Algo_BOT/dashboard_api.py.backup*` - backup files
- `store/code/IBKR_Algo_BOT/.env.corrupt.bak`
- `store/code/IBKR_Algo_BOT/ai/ai_predictor.py.backup`
- `store/code/IBKR_Algo_BOT/bridge/ib_adapter.py.*.bak`

### 9. IBKR_Algo_BOT - Install Backups (Nested recursion issue)
- `store/code/IBKR_Algo_BOT/_install_backups/20251106-200002/` - recursive nesting
- Keep only: last 3 dated backups (20251112, 20251113, most recent)

### 10. Root - Redundant Documentation
- `README.txt` - redundant with README.md
- `README_COMBO.txt` - outdated
- `README_QUICKSTART.txt` - outdated
- `README_REAL_IBKR.txt` - outdated
- `README_BACKUP.md` - backup of readme

### 11. Misc Root Files
- `package-lock.json` - not a Node.js project
- `active_tasks_autogen.json` - outdated task list
- `PING_CHECK.md` - test file

## Estimated Space to Reclaim
- Backup directories: ~500MB-1GB (recursive backups issue)
- Root old files: ~5-10MB
- IBKR_Algo_BOT backups: ~20-50MB
- Total estimated: ~520-1GB+

## Safety Notes
- Keep: Current README.md, current .gitignore, .env.example
- Keep: IBKR_Algo_BOT_V2/ (active development)
- Keep: Last 2-3 dated backups in _install_backups
- Keep: All files in src/, server/, ui/, scripts/, docs/, security/

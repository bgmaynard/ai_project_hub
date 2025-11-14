# Repository Cleanup Script
# Run with: .\cleanup_repo.ps1
# Review cleanup_plan.md before running

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Repository Cleanup Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Change to repository root
Set-Location "C:\ai_project_hub"

Write-Host "[INFO] Starting repository cleanup..." -ForegroundColor Green
Write-Host "[INFO] Current directory: $(Get-Location)" -ForegroundColor Yellow
Write-Host ""

# Create a backup summary before cleanup
$cleanupLog = "cleanup_log_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"
"Repository Cleanup - $(Get-Date)" | Out-File $cleanupLog

# Function to safely remove files
function Remove-SafelyFiles {
    param([string[]]$Files)
    foreach ($file in $Files) {
        if (Test-Path $file) {
            $size = (Get-Item $file).Length
            Write-Host "[REMOVE] $file ($('{0:N0}' -f $size) bytes)" -ForegroundColor Yellow
            "REMOVED: $file ($size bytes)" | Out-File -Append $cleanupLog
            Remove-Item $file -Force
        }
    }
}

# Function to safely remove directories
function Remove-SafelyDir {
    param([string]$Dir)
    if (Test-Path $Dir) {
        $size = (Get-ChildItem $Dir -Recurse | Measure-Object -Property Length -Sum).Sum
        Write-Host "[REMOVE DIR] $Dir ($('{0:N2}' -f ($size/1MB)) MB)" -ForegroundColor Magenta
        "REMOVED DIR: $Dir ($([math]::Round($size/1MB, 2)) MB)" | Out-File -Append $cleanupLog
        Remove-Item $Dir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "[STEP 1] Removing old backup and fixed files..." -ForegroundColor Cyan
$oldFiles = @(
    "dashboard_api_fixed.py",
    "dashboard_api_hold.py",
    "dashboard_api_immediate.py",
    "ib_adapter.py",
    "ibkr_adapter_fixed.py",
    "ai_router_complete_package.zip",
    "files.zip",
    "Run-CollabCycle.ps1.bak_20251013_200457",
    "Run-CollabCycle.ps1.bak_onlyunit_20251013_201203",
    "Run-CollabCycle.ps1.bak_rmsetcont_20251013_200609",
    "Run-CollabCycle.ps1.bak_unescape_20251013_200745"
)
Remove-SafelyFiles -Files $oldFiles

Write-Host ""
Write-Host "[STEP 2] Removing test files..." -ForegroundColor Cyan
$testFiles = @(
    "test_adapter.py",
    "test_adapter_standalone.py",
    "test_forced_id.py",
    "test_ids.py",
    "test_long.py",
    "test_tws.py",
    "test_tws_7497.py"
)
Remove-SafelyFiles -Files $testFiles

Write-Host ""
Write-Host "[STEP 3] Removing old install scripts..." -ForegroundColor Cyan
$installScripts = @(
    "INSTALL_COLOCATED_SERVER.bat",
    "INSTALL_GUI_ONLY.bat",
    "INSTALL_SERVER.bat",
    "SELF_INSTALL.bat",
    "SELF_INSTALL_COMBO.bat",
    "SELF_INSTALL_REAL_IBKR.bat"
)
Remove-SafelyFiles -Files $installScripts

Write-Host ""
Write-Host "[STEP 4] Removing old HTML files..." -ForegroundColor Cyan
$htmlFiles = @(
    "status.html",
    "trading.html",
    "trading_hub.html"
)
Remove-SafelyFiles -Files $htmlFiles

Write-Host ""
Write-Host "[STEP 5] Removing redundant scripts..." -ForegroundColor Cyan
$redundantScripts = @(
    "apply_ibkr_connection_patch.py",
    "self_installing_patch.py",
    "setup_ibkr_bot.py",
    "setup_everything.py",
    "ai_mesh_controller.py"
)
Remove-SafelyFiles -Files $redundantScripts

Write-Host ""
Write-Host "[STEP 6] Removing redundant documentation..." -ForegroundColor Cyan
$docs = @(
    "README.txt",
    "README_COMBO.txt",
    "README_QUICKSTART.txt",
    "README_REAL_IBKR.txt",
    "README_BACKUP.md",
    "package-lock.json",
    "active_tasks_autogen.json",
    "PING_CHECK.md"
)
Remove-SafelyFiles -Files $docs

Write-Host ""
Write-Host "[STEP 7] Removing problematic backup directories..." -ForegroundColor Cyan
Remove-SafelyDir -Dir "backups\deployment_backup_20251112_202939"

Write-Host ""
Write-Host "[STEP 8] Cleaning up IBKR_Algo_BOT backup files..." -ForegroundColor Cyan
if (Test-Path "store\code\IBKR_Algo_BOT") {
    Get-ChildItem "store\code\IBKR_Algo_BOT" -Recurse -Include "*.bak","*.backup*" | ForEach-Object {
        Write-Host "[REMOVE] $($_.FullName)" -ForegroundColor Yellow
        "REMOVED: $($_.FullName)" | Out-File -Append $cleanupLog
        Remove-Item $_.FullName -Force
    }
}

Write-Host ""
Write-Host "[STEP 9] Cleaning old install backups (keeping last 3)..." -ForegroundColor Cyan
if (Test-Path "store\code\IBKR_Algo_BOT\_install_backups") {
    $backupDirs = Get-ChildItem "store\code\IBKR_Algo_BOT\_install_backups" -Directory |
        Where-Object { $_.Name -match '^\d{8}-\d{6}$' } |
        Sort-Object Name -Descending

    if ($backupDirs.Count -gt 3) {
        $toRemove = $backupDirs | Select-Object -Skip 3
        foreach ($dir in $toRemove) {
            Remove-SafelyDir -Dir $dir.FullName
        }
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Cleanup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Cleanup log saved to: $cleanupLog" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Review the cleanup log" -ForegroundColor White
Write-Host "  2. Test that the system still works" -ForegroundColor White
Write-Host "  3. Commit the cleanup: git add -A && git commit -m 'chore: clean up unused files and backups'" -ForegroundColor White
Write-Host ""

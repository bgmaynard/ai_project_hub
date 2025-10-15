param(
  [string]$Root = "C:\ai_project_hub"
)

$ErrorActionPreference = "Stop"

$logName = "manual_collab_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss")
$log     = Join-Path $Root ("logs\" + $logName)

# Run the full cycle (with -Open to show outputs) and tee to a log
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Root "Run-CollabCycle.ps1") -Open *>&1 | Tee-Object -FilePath $log

# Keep a friendly pointer to today’s artifacts
$store = Join-Path $Root "store"
Write-Host "`nArtifacts:"
Write-Host "  - Graph HTML:    $(Join-Path $store 'enhanced_dependency_graph.html')"
Write-Host "  - Graph PNG:     $(Join-Path $store 'enhanced_dependency_graph.png')"
Write-Host "  - Dep Summary:   $(Join-Path $store 'dep_summary.md')"
Write-Host "  - Tasks (auto):  $(Join-Path $store 'tasks\active_tasks_autogen.json')"
Write-Host "  - Prompts:       $(Join-Path $store 'logic_docs\prompts')"
Write-Host "Log: $log"

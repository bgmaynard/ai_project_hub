param(
  [string]$DepCsv = "C:\ai_project_hub\store\dep_fanin_fanout.csv",
  [string]$OutTasks = "C:\ai_project_hub\store\tasks\active_tasks_autogen.json"
)

if (-not (Test-Path -LiteralPath $DepCsv)) {
  throw "Run Make-DepReport.ps1 first. Missing: $DepCsv"
}

$rows = Import-Csv -LiteralPath $DepCsv | ForEach-Object {
  $_.FanIn  = [int]$_.FanIn
  $_.FanOut = [int]$_.FanOut
  $_
}

$topIn  = $rows | Sort-Object FanIn  -Descending | Select-Object -First 5
$topOut = $rows | Sort-Object FanOut -Descending | Select-Object -First 5
$orph  = $rows | Where-Object { $_.FanIn -eq 0 -and $_.FanOut -eq 0 } | Select-Object -First 5

$tasks = New-Object System.Collections.Generic.List[object]

# 1) Hardening: tests around high fan-out (they influence many others)
foreach ($m in $topOut) {
  $tasks.Add([pscustomobject]@{
    task        = "Add unit tests + type checks for high fan-out module: $($m.Module)"
    rationale   = "High fan-out affects many downstream modules"
    assigned_to = "Copilot"
    cluster     = "Utilities/AI/Trading (depends on module)"
    status      = "pending"
  })
}

# 2) Refactor review: high fan-in (many depend on them)
foreach ($m in $topIn) {
  $tasks.Add([pscustomobject]@{
    task        = "Refactor / isolate interfaces for high fan-in module: $($m.Module)"
    rationale   = "Reduce coupling; create stable interface to protect dependents"
    assigned_to = "ChatGPT"
    cluster     = "Depends on module"
    status      = "pending"
  })
}

# 3) Cleanup: orphans
foreach ($m in $orph) {
  $tasks.Add([pscustomobject]@{
    task        = "Classify orphan module: $($m.Module) (delete, test-only, or wire into flow)"
    rationale   = "Remove dead code or make intent explicit"
    assigned_to = "Claude"
    cluster     = "Triage"
    status      = "pending"
  })
}

$tasks | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 -LiteralPath $OutTasks
Write-Host "Wrote: $OutTasks"

# =====================================================================
# Initialize-AIProjectHub.ps1  (safe, robust, copy-paste version)
# Builds a logic layout + inventory for your AI project
# =====================================================================

param(
  [string]$Root = "C:\ai_project_hub",
  [string]$ProjectPath = "C:\ai_project_hub\store\code\IBKR_Algo_BOT"
)

$ErrorActionPreference = "Stop"

# --------------------- helpers ---------------------
function Ensure-Folder($path) {
  if (-not (Test-Path -LiteralPath $path)) {
    New-Item -ItemType Directory -Path $path | Out-Null
  }
}

function Get-Lang($ext) {
  switch -Regex ($ext.ToLower()) {
    '\.py$'               { "python" ; break }
    '\.md$'               { "markdown"; break }
    '\.txt$'              { "text"; break }
    '\.js$'               { "javascript"; break }
    '\.ts$'               { "typescript"; break }
    '\.tsx$'              { "tsx"; break }
    '\.jsx$'              { "jsx"; break }
    '\.json$'             { "json"; break }
    '\.yml$|\.yaml$'      { "yaml"; break }
    default               { ($ext.TrimStart('.')).ToLower() }
  }
}

function Read-TextSafe([string]$filePath) {
  try {
    return (Get-Content -LiteralPath $filePath -Raw -ErrorAction Stop)
  } catch {
    try {
      $bytes = [System.IO.File]::ReadAllBytes($filePath)
      if ($bytes.Length -gt 0) {
        # try UTF8 first, if decode fails return empty (we don't regex on it)
        return [System.Text.Encoding]::UTF8.GetString($bytes)
      } else {
        return ""
      }
    } catch {
      return ""
    }
  }
}

# Skip folders commonly not part of your codebase
$SkipDirs = @(
  "\venv\", "\.venv\", "\__pycache__\",
  "\.git\", "\node_modules\", "\dist\", "\build\"
)

function Should-Skip([string]$fullPath) {
  foreach ($sd in $SkipDirs) {
    if ($fullPath -replace '/','\' -like "*$sd*") { return $true }
  }
  return $false
}

# --------------------- ensure structure ---------------------
$store = Join-Path $Root "store"
$codeDir  = Join-Path $store "code"
$logicDir = Join-Path $store "logic_docs"
$tasksDir = Join-Path $store "tasks"

Ensure-Folder $Root
Ensure-Folder $store
Ensure-Folder $codeDir
Ensure-Folder $logicDir
Ensure-Folder $tasksDir

if (-not (Test-Path -LiteralPath $ProjectPath)) {
  throw "Project path not found: $ProjectPath"
}

# --------------------- scan files ---------------------
Write-Host "Scanning project files under: $ProjectPath"

$allFiles = Get-ChildItem -Path $ProjectPath -Recurse -File -ErrorAction Stop `
  | Where-Object { -not (Should-Skip $_.FullName) }

$files = $allFiles | ForEach-Object {
  [PSCustomObject]@{
    path            = $_.FullName
    rel_path        = $_.FullName.Substring($ProjectPath.Length).TrimStart('\','/')
    name            = $_.Name
    ext             = $_.Extension
    language        = Get-Lang($_.Extension)
    size_bytes      = $_.Length
    last_write_time = $_.LastWriteTimeUtc.ToString("o")
  }
}

# --------------------- language stats ---------------------
$langStats = $files | Group-Object language | ForEach-Object {
  [PSCustomObject]@{
    language = $_.Name
    files    = $_.Count
    bytes    = ($_.Group | Measure-Object size_bytes -Sum).Sum
  }
}

# --------------------- python import graph ---------------------
Write-Host "Building Python import dependency graph..."

$pyFiles = $files | Where-Object { $_.language -eq "python" }
$importEdges = New-Object System.Collections.Generic.List[object]
$moduleMap   = @{}

foreach ($f in $pyFiles) {
  $text = Read-TextSafe $f.path

  # still register module even if unreadable
  $rel  = $f.rel_path -replace '\\','/'
  $modName = ($rel -replace '\.py$','') -replace '/','.'
  $moduleMap[$modName] = $rel

  if ([string]::IsNullOrWhiteSpace($text)) { continue }

  # Match `import pkg.sub` and `from pkg.sub import X`
  $imports = @()
  $imports += [regex]::Matches($text, '^[\s]*import\s+([A-Za-z0-9_\.]+)', 'Multiline') `
             | ForEach-Object { $_.Groups[1].Value }
  $imports += [regex]::Matches($text, '^[\s]*from\s+([A-Za-z0-9_\.]+)\s+import\s+', 'Multiline') `
             | ForEach-Object { $_.Groups[1].Value }

  $imports = $imports | Where-Object { $_ -and $_.Trim().Length -gt 0 } | Select-Object -Unique
  foreach ($imp in $imports) {
    $importEdges.Add([PSCustomObject]@{
      from = $modName
      to   = $imp
      file = $rel
    })
  }
}

# Normalize edges to local modules when possible
$localModules = $moduleMap.Keys
$edgesNormalized = foreach ($e in $importEdges) {
  $target = $e.to
  if (-not $target) { continue }
  $parts = $target.Split('.')
  while ($parts.Count -gt 0) {
    $candidate = ($parts -join '.')
    if ($localModules -contains $candidate) { $target = $candidate; break }
    if ($parts.Count -le 1) { break }
    $parts = $parts[0..($parts.Count-2)]
  }
  [PSCustomObject]@{ from=$e.from; to=$target; file=$e.file }
}

# Write Graphviz DOT
$dotPath = Join-Path $store "dependency_graph.dot"
$dot = "digraph G {`n  rankdir=LR;`n  node [shape=box, fontsize=10];"
$edgesOut = $edgesNormalized | Select-Object -Unique from,to `
            | Where-Object { $_.from -and $_.to }
foreach ($edge in $edgesOut) {
  $from = $edge.from -replace '"','\"'
  $to   = $edge.to   -replace '"','\"'
  $dot += "`n  `"$from`" -> `"$to`";"
}
$dot += "`n}"
Set-Content -LiteralPath $dotPath -Value $dot -Encoding UTF8

# --------------------- write project summary ---------------------
$summaryPath = Join-Path $store "project_summary.json"
$now = (Get-Date).ToUniversalTime().ToString("o")

$summary = [ordered]@{
  project_name = "IBKR_Algo_BOT"
  description  = "Unified AI trading bot combining LSTM/MTF, Warrior-style momentum, and IBKR execution."
  created_or_updated_utc = $now
  location     = $ProjectPath
  inventory    = $files
  language_stats = $langStats
  dependency_graph = @{
    type  = "python_imports"
    nodes = ($moduleMap.GetEnumerator() | ForEach-Object { @{ module=$_.Key; file=$_.Value } })
    edges = $edgesNormalized
    graphviz_dot = $dotPath
  }
  linked_ai_projects = @{
    claude_project  = @{ path = $ProjectPath; role="logic_designer"; status="active" }
    chatgpt_project = @{ path = $store; role="orchestrator"; status="active" }
    copilot_project = @{ path = $ProjectPath; role="developer"; status="active" }
  }
  active_tasks_file    = (Join-Path $tasksDir "active_tasks.json")
  completed_tasks_file = (Join-Path $tasksDir "completed_tasks.json")
  changelog            = (Join-Path $store "changelog.md")
}

$summary | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $summaryPath -Encoding UTF8

# --------------------- init tasks & changelog ---------------------
$activeTasksPath    = Join-Path $tasksDir "active_tasks.json"
$completedTasksPath = Join-Path $tasksDir "completed_tasks.json"
$changelogPath      = Join-Path $store "changelog.md"

if (-not (Test-Path -LiteralPath $activeTasksPath)) {
  @(
    @{
      task        = "Inventory and dependency mapping for IBKR_Algo_BOT"
      assigned_to = "ChatGPT"
      status      = "done"
    },
    @{
      task        = "Add adaptive trailing stops (ATR, momentum score) to risk module"
      assigned_to = "Copilot"
      status      = "pending"
    },
    @{
      task        = "Summarize strategy notes into logic_docs/mtf_logic.md"
      assigned_to = "Claude"
      status      = "pending"
    }
  ) | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $activeTasksPath -Encoding UTF8
}

if (-not (Test-Path -LiteralPath $completedTasksPath)) {
  "[]" | Set-Content -LiteralPath $completedTasksPath -Encoding UTF8
}

if (-not (Test-Path -LiteralPath $changelogPath)) {
  "# Changelog`n`n[$now] Initialized logic layout, inventory, and dependency graph." `
    | Set-Content -LiteralPath $changelogPath -Encoding UTF8
}

# --------------------- drop minimal orchestrator if missing ---------------------
$orchPath = Join-Path $Root "orchestrator.py"
if (-not (Test-Path -LiteralPath $orchPath)) {
$orch = @'
import json, os, datetime

STORE = os.path.join(r"C:\ai_project_hub","store","project_summary.json")
CHANGELOG = os.path.join(r"C:\ai_project_hub","store","changelog.md")

def load_summary():
    with open(STORE,"r",encoding="utf-8") as f:
        return json.load(f)

def save_summary(data):
    with open(STORE,"w",encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def log(msg):
    with open(CHANGELOG,"a",encoding="utf-8") as f:
        f.write(f"[{datetime.datetime.utcnow().isoformat()}Z] {msg}\n")

if __name__ == "__main__":
    import sys
    task = " ".join(sys.argv[1:]) or "no-op"
    s = load_summary()
    s["last_orchestrator_task"] = {
        "when": datetime.datetime.utcnow().isoformat()+"Z",
        "task": task
    }
    save_summary(s)
    log(f"orchestrator: {task}")
    print("OK:", task)
'@
  Set-Content -LiteralPath $orchPath -Value $orch -Encoding UTF8
}

# --------------------- footer ---------------------
Write-Host ""
Write-Host "✅ Logic layout built."
Write-Host ("  - Summary:     {0}" -f $summaryPath)
Write-Host ("  - Tasks:       {0}" -f $activeTasksPath)
Write-Host ("  - Changelog:   {0}" -f $changelogPath)
Write-Host ("  - Graphviz:    {0}" -f $dotPath)
Write-Host ""

# Show a safe Graphviz command without breaking quotes
$pngOut = ($dotPath -replace '\.dot$','.png')
Write-Host "Tip: Render the DOT file with Graphviz:"
Write-Host ("     dot -Tpng `"{0}`" -o `"{1}`"" -f $dotPath, $pngOut) -ForegroundColor Yellow

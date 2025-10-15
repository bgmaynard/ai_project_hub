# Make-DepReport.ps1  — robust fan-in / fan-out + markdown summary
param(
  [string]$DotPath = "C:\ai_project_hub\store\enhanced_dependency_graph.dot",
  [string]$OutCsv  = "C:\ai_project_hub\store\dep_fanin_fanout.csv",
  [string]$OutMd   = "C:\ai_project_hub\store\dep_summary.md"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $DotPath)) {
  throw "DOT not found: $DotPath"
}

# Parse edges: "A" -> "B"
$dot = Get-Content -LiteralPath $DotPath -Raw
$edges = [regex]::Matches($dot, '^\s*"(.*?)"\s*->\s*"(.*?)"', 'Multiline') | ForEach-Object {
  [pscustomobject]@{ From = $_.Groups[1].Value; To = $_.Groups[2].Value }
}

# Build node set + counts
$allNodes = New-Object System.Collections.Generic.HashSet[string]
$fanIn  = @{}  # node -> int
$fanOut = @{}  # node -> int

foreach ($e in $edges) {
  $allNodes.Add($e.From) | Out-Null
  $allNodes.Add($e.To)   | Out-Null

  if (-not $fanOut.ContainsKey($e.From)) { $fanOut[$e.From] = 0 }
  if (-not $fanIn.ContainsKey($e.To))    { $fanIn[$e.To]    = 0 }
  $fanOut[$e.From] += 1
  $fanIn[$e.To]    += 1
}

# Ensure every node has 0 if missing
foreach ($n in $allNodes) {
  if (-not $fanIn.ContainsKey($n))  { $fanIn[$n]  = 0 }
  if (-not $fanOut.ContainsKey($n)) { $fanOut[$n] = 0 }
}

# Build report objects
$report = $allNodes | ForEach-Object {
  [pscustomobject]@{
    Module = $_
    FanIn  = [int]$fanIn[$_]
    FanOut = [int]$fanOut[$_]
  }
}

# Sort by FanIn desc, then FanOut desc (correct syntax)
$sorted = $report | Sort-Object `
  @{Expression='FanIn';Descending=$true}, `
  @{Expression='FanOut';Descending=$true}

# Save CSV
$sorted | Export-Csv -NoTypeInformation -Path $OutCsv -Encoding UTF8

# Top lists + orphans
$topIn   = $sorted | Select-Object -First 8
$topOut  = $report | Sort-Object @{Expression='FanOut';Descending=$true}, @{Expression='FanIn';Descending=$true} | Select-Object -First 8
$orphans = $report | Where-Object { $_.FanIn -eq 0 -and $_.FanOut -eq 0 }

# Markdown summary (ASCII-only)
$md = @()
$md += "# Dependency Summary"
$md += ""
$md += "CSV: `$OutCsv`"
$md += ""
$md += "## Top by Fan-In (most depended-on)"
foreach ($r in $topIn)  { $md += "* $($r.Module) - FanIn=$($r.FanIn), FanOut=$($r.FanOut)" }
$md += ""
$md += "## Top by Fan-Out (depends on many)"
foreach ($r in $topOut) { $md += "* $($r.Module) - FanIn=$($r.FanIn), FanOut=$($r.FanOut)" }
$md += ""
$md += "## Orphans (no incoming or outgoing edges)"
if ($orphans) {
  foreach ($r in $orphans) { $md += "* $($r.Module)" }
} else {
  $md += "* (none)"
}

# Write Markdown
$md -join "`r`n" | Set-Content -LiteralPath $OutMd -Encoding UTF8

Write-Host "Saved:"
Write-Host "  $OutCsv"
Write-Host "  $OutMd"

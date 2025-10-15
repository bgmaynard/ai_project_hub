param(
  [string]$Root = "C:\ai_project_hub",
  [switch]$Open
)

$ErrorActionPreference = "Stop"

# -------- paths --------
$store       = Join-Path $Root "store"
$summaryPath = Join-Path $store "project_summary.json"
$enhDot      = Join-Path $store "enhanced_dependency_graph.dot"
$enhPng      = Join-Path $store "enhanced_dependency_graph.png"
$htmlPath    = Join-Path $store "enhanced_dependency_graph.html"

if (-not (Test-Path -LiteralPath $summaryPath)) {
  throw "Not found: $summaryPath. Run Initialize-AIProjectHub.ps1 first."
}

# --- helper: write UTF-8 *without* BOM ---
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
function Set-ContentUtf8NoBom([string]$Path, [string]$Text) {
  [System.IO.File]::WriteAllText($Path, $Text, $Utf8NoBom)
}

# --- load summary (nodes/edges) ---
$summary = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
$dep     = $summary.dependency_graph
$nodes   = @{}; foreach ($n in $dep.nodes) { $nodes[$n.module] = $n.file }
$edges   = @(); foreach ($e in $dep.edges) { if ($e.from -and $e.to) { $edges += [pscustomobject]@{from=$e.from;to=$e.to;file=$e.file} } }

# --- grouping rules ---
function Get-Group([string]$module) {
  $m = $module.ToLower()
  if ($m -match 'ibkr|ibapi|totalview|order|execution|contract|client|wrapper|trading_backend|reconcil') { return "Trading" }
  if ($m -match 'dashboard|api|socket|flask|fastapi|server|ui|web|modular_dashboard|config') { return "Dashboard" }
  if ($m -match 'lstm|trainer|training|mtf|model|backtest|feature|xgboost|scikit|tensorflow|torch|predict|inference') { return "AI" }
  if ($m -match 'util|common|helper|logger|config|settings|constants|tools') { return "Utilities" }
  return "Utilities"
}

# local/external split
$localSet = [System.Collections.Generic.HashSet[string]]::new()
$nodes.Keys | ForEach-Object { [void]$localSet.Add($_) }
function Is-Local([string]$name) { $localSet.Contains($name) }

# node meta (include externals referenced by edges)
$nodeMeta = @{}
foreach ($name in $nodes.Keys) { $nodeMeta[$name] = [pscustomobject]@{ group=(Get-Group $name); local=$true } }
foreach ($e in $edges) {
  if (-not $nodeMeta.ContainsKey($e.to) -and -not (Is-Local $e.to)) {
    $nodeMeta[$e.to] = [pscustomobject]@{ group="External"; local=$false }
  }
}

# styles
$styles = @{
  "Trading"   = @{ fill="#E6F3FF"; border="#2B6CB0" }
  "AI"        = @{ fill="#FFF5E6"; border="#C05621" }
  "Dashboard" = @{ fill="#E6FFFA"; border="#2C7A7B" }
  "Utilities" = @{ fill="#F3E8FF"; border="#6B46C1" }
  "External"  = @{ fill="#F7FAFC"; border="#4A5568" }
}

# --- build DOT text (clustered, colored) ---
$sb = New-Object System.Text.StringBuilder
[void]$sb.AppendLine("digraph G {")
[void]$sb.AppendLine("  rankdir=LR;")
[void]$sb.AppendLine("  graph [fontname=""Segoe UI"", fontsize=10, splines=true, overlap=false, concentrate=true];")
[void]$sb.AppendLine("  node  [shape=box, style=""rounded,filled"", fontname=""Segoe UI"", fontsize=10];")
[void]$sb.AppendLine("  edge  [fontname=""Segoe UI"", fontsize=8, color=""#718096"", arrowsize=0.7];")

$groups = @("Trading","AI","Dashboard","Utilities","External")
foreach ($g in $groups) {
  $fill = $styles[$g].fill; $border = $styles[$g].border
  [void]$sb.AppendLine("  subgraph cluster_$($g.ToLower()) {")
  [void]$sb.AppendLine("    label=""$g""; color=""$border""; style=""rounded""; bgcolor=""$fill"";")

  $groupNodes = $nodeMeta.GetEnumerator() | Where-Object { $_.Value.group -eq $g } | Select-Object -ExpandProperty Key
  foreach ($n in $groupNodes) {
    if ($g -eq "External") {
      [void]$sb.AppendLine("    `"$n`" [color=""$border"", penwidth=1, style=""dashed,filled""];")
    } else {
      [void]$sb.AppendLine("    `"$n`" [color=""$border"", penwidth=1.4];")
    }
  }
  [void]$sb.AppendLine("  }")
}

$edgeSet = New-Object System.Collections.Generic.HashSet[string]
foreach ($e in $edges) {
  $from = $e.from; $to = $e.to
  if (-not $from -or -not $to) { continue }
  $key = "$from=>$to"; if ($edgeSet.Contains($key)) { continue } [void]$edgeSet.Add($key)
  $edgeColor = if ($nodeMeta[$to] -and -not $nodeMeta[$to].local) { "#A0AEC0" } else { "#4A5568" }
  [void]$sb.AppendLine("  `"$from`" -> `"$to`" [color=""$edgeColor""];")
}

[void]$sb.AppendLine("}")
$dotText = $sb.ToString()

# --- write DOT without BOM ---
Set-ContentUtf8NoBom -Path $enhDot -Text $dotText
Write-Host ("Wrote: {0}" -f $enhDot)

# --- render PNG if dot.exe available ---
$dotExe = "C:\Program Files\Graphviz\bin\dot.exe"
if (-not (Test-Path -LiteralPath $dotExe)) {
  $probe = Get-ChildItem "C:\Program Files","C:\Program Files (x86)","$env:LocalAppData\Programs" -Recurse -Filter dot.exe -ErrorAction SilentlyContinue |
           Where-Object { $_.FullName -match 'graphviz[\\\/]bin[\\\/]dot\.exe' } | Select-Object -First 1 -Expand FullName
  if ($probe) { $dotExe = $probe }
}

if (Test-Path -LiteralPath $dotExe) {
  & $dotExe -Tpng $enhDot -o $enhPng | Out-Null
  if (Test-Path -LiteralPath $enhPng) {
    Write-Host ("Rendered: {0}" -f $enhPng)
    if ($Open) { Start-Process $enhPng }
  }
} else {
  Write-Host "Graphviz dot.exe not found; skipping PNG render. Install via winget: winget install --id Graphviz.Graphviz -e"
}

# --- HTML (no escaping): page fetches DOT at runtime and renders with Viz.js ---
$dotRel = [IO.Path]::GetFileName($enhDot)
$html = @"
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Enhanced Dependency Graph</title>
<style>html,body,#container{height:100%;margin:0;font-family:Segoe UI,Arial}</style>
</head>
<body>
<div id="container">Loading graph…</div>
<script src="https://cdn.jsdelivr.net/npm/viz.js@2.1.2/viz.js"></script>
<script src="https://cdn.jsdelivr.net/npm/viz.js@2.1.2/full.render.js"></script>
<script>
fetch("$dotRel").then(r => r.text()).then(dot => {
  const viz = new Viz();
  return viz.renderSVGElement(dot);
}).then(svg => {
  const c = document.getElementById('container');
  c.innerHTML = "";
  c.appendChild(svg);
}).catch(err => {
  document.getElementById('container').textContent = err;
});
</script>
</body>
</html>
"@

# write HTML without BOM
Set-ContentUtf8NoBom -Path $htmlPath -Text $html
Write-Host ("HTML viewer: {0}" -f $htmlPath)
if ($Open) { Start-Process $htmlPath }

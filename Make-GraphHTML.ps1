param(
  [string]$DotPath = "C:\ai_project_hub\store\enhanced_dependency_graph.dot",
  [string]$HtmlOut = "C:\ai_project_hub\store\enhanced_dependency_graph.html"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $DotPath)) { throw "DOT not found: $DotPath" }

# Read DOT (handles BOM fine), write HTML as UTF-8 no BOM
$dot = Get-Content -LiteralPath $DotPath -Raw
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
function Write-NoBom($path,$text){ [System.IO.File]::WriteAllText($path,$text,$Utf8NoBom) }

# Put DOT inside a <script type="text/plain"> to avoid escaping headaches & fetch()
# Viz.js will read it from the DOM and render.
$html = @"
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Enhanced Dependency Graph</title>
<style>html,body,#container{height:100%;margin:0;font-family:Segoe UI,Arial}</style>
</head>
<body>
<div id="container">Rendering…</div>
<script id="dot-src" type="text/plain">
@DOT@
</script>
<script src="https://cdn.jsdelivr.net/npm/viz.js@2.1.2/viz.js"></script>
<script src="https://cdn.jsdelivr.net/npm/viz.js@2.1.2/full.render.js"></script>
<script>
const dot = document.getElementById('dot-src').textContent;
const viz = new Viz();
viz.renderSVGElement(dot).then(svg => {
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

# Avoid accidentally closing the <script> by ensuring no "</script>" appears in DOT
$dotSafe = $dot -replace '</script>','</scr"+"ipt>'
$htmlOutText = $html -replace '@DOT@', $dotSafe

Write-NoBom $HtmlOut $htmlOutText
Write-Host "HTML viewer written to: $HtmlOut"

# Open it
Start-Process $HtmlOut

param(
  [string]$Message = "cont: routine push",
  [switch]$All = $false
)

$ErrorActionPreference = "Stop"
$repo = "C:\ai_project_hub"
Set-Location $repo

git status

if ($All) {
  git add -A
} else {
  git add store/code/IBKR_Algo_BOT/dashboard_api.py `
          store/code/IBKR_Algo_BOT/ui/index.html `
          store/code/IBKR_Algo_BOT/ui/app.js `
          store/code/IBKR_Algo_BOT/ui/styles.css `
          store/ai_shared/mesh_heartbeat.json `
          .github `
          CODEOWNERS `
          README_COLLAB.md
}

git commit -m $Message
git push
\# Simple IBKR Connection Fix - Step by Step

\# Run each command separately in PowerShell



\# Step 1: Create scripts directory

New-Item -Path "scripts" -ItemType Directory -Force



\# Step 2: Install required packages

pip install --upgrade python-dotenv ib-insync fastapi uvicorn\[standard]



\# Step 3: Create .env file (if it doesn't exist)

if (-not (Test-Path ".env")) {

&nbsp;   @"

LOCAL\_API\_KEY=My\_Super\_Strong\_Key\_123

API\_HOST=127.0.0.1

API\_PORT=9101

TWS\_HOST=127.0.0.1

TWS\_PORT=7497

TWS\_CLIENT\_ID=6001

IB\_HEARTBEAT\_SEC=3.0

IB\_CONNECT\_TIMEOUT\_SEC=15.0

"@ | Set-Content ".env" -Encoding UTF8

&nbsp;   Write-Host "✅ Created .env file" -ForegroundColor Green

}



\# Step 4: Create diagnostic script

@"

import os

import socket

from pathlib import Path



def main():

&nbsp;   print("🔍 IBKR Connection Diagnostics")

&nbsp;   print("=" \* 40)

&nbsp;   

&nbsp;   # Load environment

&nbsp;   try:

&nbsp;       from dotenv import load\_dotenv

&nbsp;       load\_dotenv()

&nbsp;       print("✅ Environment loaded")

&nbsp;   except ImportError:

&nbsp;       print("❌ python-dotenv not installed")

&nbsp;       return

&nbsp;   

&nbsp;   # Check environment variables

&nbsp;   print("\\n🔧 Environment Variables:")

&nbsp;   for var in \["TWS\_HOST", "TWS\_PORT", "TWS\_CLIENT\_ID", "LOCAL\_API\_KEY"]:

&nbsp;       value = os.getenv(var)

&nbsp;       display = "\*\*\*" if "KEY" in var else value

&nbsp;       status = "✅" if value else "❌"

&nbsp;       print(f"   {status} {var}={display}")

&nbsp;   

&nbsp;   # Test TWS connection

&nbsp;   host = os.getenv("TWS\_HOST", "127.0.0.1")

&nbsp;   port = int(os.getenv("TWS\_PORT", "7497"))

&nbsp;   

&nbsp;   print(f"\\n🔗 Testing connection to {host}:{port}")

&nbsp;   try:

&nbsp;       with socket.create\_connection((host, port), timeout=5):

&nbsp;           print("✅ TWS is listening and accepting connections")

&nbsp;   except ConnectionRefusedError:

&nbsp;       print("❌ Connection refused - TWS not listening")

&nbsp;       print("💡 Start TWS/Gateway and enable API")

&nbsp;   except Exception as e:

&nbsp;       print(f"❌ Connection failed: {e}")

&nbsp;   

&nbsp;   # Test imports

&nbsp;   print("\\n📦 Testing imports:")

&nbsp;   try:

&nbsp;       import ib\_insync

&nbsp;       print("✅ ib\_insync available")

&nbsp;   except ImportError:

&nbsp;       print("❌ ib\_insync not installed")

&nbsp;   

&nbsp;   try:

&nbsp;       import fastapi

&nbsp;       print("✅ fastapi available")

&nbsp;   except ImportError:

&nbsp;       print("❌ fastapi not installed")



if \_\_name\_\_ == "\_\_main\_\_":

&nbsp;   main()

"@ | Set-Content "scripts\\diagnose\_connection.py" -Encoding UTF8



Write-Host "✅ Created diagnostic script" -ForegroundColor Green



\# Step 5: Create simple bootstrap script

@"

import subprocess

import sys

import os

from pathlib import Path



def main():

&nbsp;   print("🚀 Starting IBKR Trading Bot...")

&nbsp;   

&nbsp;   # Load environment

&nbsp;   try:

&nbsp;       from dotenv import load\_dotenv

&nbsp;       load\_dotenv()

&nbsp;       print("✅ Environment loaded")

&nbsp;   except ImportError:

&nbsp;       print("❌ Installing python-dotenv...")

&nbsp;       subprocess.run(\[sys.executable, "-m", "pip", "install", "python-dotenv"])

&nbsp;       from dotenv import load\_dotenv

&nbsp;       load\_dotenv()

&nbsp;   

&nbsp;   # Check TWS connection

&nbsp;   import socket

&nbsp;   host = os.getenv("TWS\_HOST", "127.0.0.1")

&nbsp;   port = int(os.getenv("TWS\_PORT", "7497"))

&nbsp;   

&nbsp;   try:

&nbsp;       with socket.create\_connection((host, port), timeout=3):

&nbsp;           print(f"✅ TWS detected on {host}:{port}")

&nbsp;   except:

&nbsp;       print(f"❌ TWS not detected on {host}:{port}")

&nbsp;       print("💡 Start TWS/Gateway and enable API settings")

&nbsp;   

&nbsp;   # Start API

&nbsp;   api\_path = Path("store/code/IBKR\_Algo\_BOT/dashboard\_api.py")

&nbsp;   if api\_path.exists():

&nbsp;       print("🚀 Starting dashboard API...")

&nbsp;       os.chdir("store/code/IBKR\_Algo\_BOT")

&nbsp;       subprocess.run(\[sys.executable, "dashboard\_api.py"])

&nbsp;   else:

&nbsp;       print("❌ dashboard\_api.py not found")



if \_\_name\_\_ == "\_\_main\_\_":

&nbsp;   main()

"@ | Set-Content "scripts\\start\_bot.py" -Encoding UTF8



Write-Host "✅ Created startup script" -ForegroundColor Green



\# Step 6: Fix the main issue - environment loading in ib\_adapter.py

$adapterPath = "store\\code\\IBKR\_Algo\_BOT\\bridge\\ib\_adapter.py"

if (Test-Path $adapterPath) {

&nbsp;   $content = Get-Content $adapterPath -Raw

&nbsp;   

&nbsp;   # Check if already patched

&nbsp;   if ($content -notmatch "load\_dotenv") {

&nbsp;       # Add environment loading at the top

&nbsp;       $envFix = @"

\# CRITICAL FIX: Load environment before any configuration

import os

import sys

from pathlib import Path



\# Navigate to project root and load .env

project\_root = Path(\_\_file\_\_).parent.parent.parent.parent

sys.path.insert(0, str(project\_root))



try:

&nbsp;   from dotenv import load\_dotenv

&nbsp;   load\_dotenv(project\_root / '.env')

&nbsp;   print(f"✅ Loaded .env from {project\_root / '.env'}")

except ImportError:

&nbsp;   print("⚠️ python-dotenv not installed")



\# Original imports continue below...

"@

&nbsp;       

&nbsp;       $newContent = $envFix + "`n" + $content

&nbsp;       $newContent | Set-Content $adapterPath -Encoding UTF8

&nbsp;       Write-Host "✅ Patched ib\_adapter.py with environment loading" -ForegroundColor Green

&nbsp;   } else {

&nbsp;       Write-Host "✅ ib\_adapter.py already patched" -ForegroundColor Green

&nbsp;   }

} else {

&nbsp;   Write-Host "⚠️ ib\_adapter.py not found at $adapterPath" -ForegroundColor Yellow

}



Write-Host ""

Write-Host "🎉 Simple fix complete!" -ForegroundColor Green

Write-Host ""

Write-Host "📋 Next steps:" -ForegroundColor Cyan

Write-Host "1. python scripts\\diagnose\_connection.py" -ForegroundColor White

Write-Host "2. python scripts\\start\_bot.py" -ForegroundColor White

Write-Host ""

Write-Host "🔧 Before starting, ensure:" -ForegroundColor Yellow

Write-Host "  - TWS or IB Gateway is running" -ForegroundColor White

Write-Host "  - API is enabled in TWS settings" -ForegroundColor White

Write-Host "  - Socket port is set to 7497" -ForegroundColor White




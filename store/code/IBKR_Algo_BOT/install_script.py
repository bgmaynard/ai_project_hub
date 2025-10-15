"""
Trading Dashboard - One-Click Setup Script
Run this to automatically set up your dashboard
"""

import os
import sys
import subprocess
from pathlib import Path
import json

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")

def print_step(step, text):
    """Print step"""
    print(f"[{step}] {text}")

def run_command(cmd, shell=True):
    """Run command and return success status"""
    try:
        result = subprocess.run(cmd, shell=shell, check=True, 
                              capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def check_python_packages():
    """Check if required Python packages are installed"""
    print_step("✓", "Checking Python packages...")
    
    required = ['flask', 'flask_cors', 'flask_socketio']
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"  ✓ {package} installed")
        except ImportError:
            missing.append(package)
            print(f"  ✗ {package} MISSING")
    
    return missing

def install_python_packages(packages):
    """Install missing Python packages"""
    print_step("⚙", f"Installing {len(packages)} Python packages...")
    
    cmd = f"{sys.executable} -m pip install " + " ".join(packages)
    success, output = run_command(cmd)
    
    if success:
        print("  ✓ All packages installed successfully")
        return True
    else:
        print(f"  ✗ Installation failed: {output}")
        return False

def check_node():
    """Check if Node.js is installed"""
    print_step("✓", "Checking Node.js installation...")
    
    success, output = run_command("node --version")
    if success:
        version = output.strip()
        print(f"  ✓ Node.js {version} installed")
        return True
    else:
        print("  ✗ Node.js NOT installed")
        print("\n  Please install Node.js from: https://nodejs.org/")
        print("  Download the LTS (Long Term Support) version")
        return False

def create_directory_structure():
    """Create project directories"""
    print_step("📁", "Creating directory structure...")
    
    dirs = [
        'dashboard_data',
        'dashboard_data/watchlists',
        'dashboard_data/exports',
        'frontend',
        'frontend/src',
        'backups'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Created {dir_path}")
    
    return True

def create_frontend_files():
    """Create frontend configuration files"""
    print_step("⚙", "Creating frontend configuration...")
    
    # package.json
    package_json = {
        "name": "trading-dashboard",
        "private": True,
        "version": "1.0.0",
        "type": "module",
        "scripts": {
            "dev": "vite",
            "build": "vite build",
            "preview": "vite preview"
        },
        "dependencies": {
            "react": "^18.2.0",
            "react-dom": "^18.2.0",
            "lucide-react": "^0.263.1"
        },
        "devDependencies": {
            "@types/react": "^18.2.15",
            "@types/react-dom": "^18.2.7",
            "@vitejs/plugin-react": "^4.0.3",
            "autoprefixer": "^10.4.14",
            "postcss": "^8.4.27",
            "tailwindcss": "^3.3.3",
            "vite": "^4.4.5"
        }
    }
    
    with open('frontend/package.json', 'w') as f:
        json.dump(package_json, f, indent=2)
    print("  ✓ Created package.json")
    
    # vite.config.js
    vite_config = """import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:5000',
        changeOrigin: true
      },
      '/socket.io': {
        target: 'http://localhost:5000',
        changeOrigin: true,
        ws: true
      }
    }
  }
})
"""
    
    with open('frontend/vite.config.js', 'w') as f:
        f.write(vite_config)
    print("  ✓ Created vite.config.js")
    
    # tailwind.config.js
    tailwind_config = """export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
"""
    
    with open('frontend/tailwind.config.js', 'w') as f:
        f.write(tailwind_config)
    print("  ✓ Created tailwind.config.js")
    
    # postcss.config.js
    postcss_config = """export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
"""
    
    with open('frontend/postcss.config.js', 'w') as f:
        f.write(postcss_config)
    print("  ✓ Created postcss.config.js")
    
    # index.html
    index_html = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/vite.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Trading Bot Dashboard</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
"""
    
    with open('frontend/index.html', 'w') as f:
        f.write(index_html)
    print("  ✓ Created index.html")
    
    # main.jsx
    main_jsx = """import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
"""
    
    with open('frontend/src/main.jsx', 'w') as f:
        f.write(main_jsx)
    print("  ✓ Created main.jsx")
    
    # index.css
    index_css = """@tailwind base;
@tailwind components;
@tailwind utilities;

body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
    sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
"""
    
    with open('frontend/src/index.css', 'w') as f:
        f.write(index_css)
    print("  ✓ Created index.css")
    
    return True

def install_frontend_dependencies():
    """Install frontend dependencies"""
    print_step("📦", "Installing frontend dependencies (this may take a few minutes)...")
    
    os.chdir('frontend')
    success, output = run_command("npm install")
    os.chdir('..')
    
    if success:
        print("  ✓ Frontend dependencies installed")
        return True
    else:
        print(f"  ✗ Installation failed: {output}")
        return False

def create_start_scripts():
    """Create convenient start scripts"""
    print_step("📝", "Creating start scripts...")
    
    # Windows batch file
    start_backend_bat = """@echo off
echo Starting Trading Dashboard Backend...
python dashboard_api.py
pause
"""
    
    with open('start_backend.bat', 'w') as f:
        f.write(start_backend_bat)
    print("  ✓ Created start_backend.bat")
    
    # Windows batch file for frontend
    start_frontend_bat = """@echo off
echo Starting Trading Dashboard Frontend...
cd frontend
npm run dev
pause
"""
    
    with open('start_frontend.bat', 'w') as f:
        f.write(start_frontend_bat)
    print("  ✓ Created start_frontend.bat")
    
    # Combined start script
    start_all_bat = """@echo off
echo Starting Trading Dashboard...
echo.
echo Starting Backend...
start cmd /k "python dashboard_api.py"
timeout /t 3 /nobreak >nul
echo.
echo Starting Frontend...
start cmd /k "cd frontend && npm run dev"
echo.
echo Dashboard will open at http://localhost:3000
echo Backend API at http://localhost:5000
echo.
echo Press any key to stop all services...
pause >nul
taskkill /F /FI "WINDOWTITLE eq Trading Dashboard*"
"""
    
    with open('start_dashboard.bat', 'w') as f:
        f.write(start_all_bat)
    print("  ✓ Created start_dashboard.bat")
    
    return True

def print_next_steps():
    """Print next steps for user"""
    print_header("✅ Setup Complete!")
    
    print("""
Your trading dashboard is ready! Here's what to do next:

📋 NEXT STEPS:

1. Copy the React dashboard code (provided separately) into:
   → frontend/src/App.jsx

2. Start the dashboard:
   Option A: Run start_dashboard.bat (starts both backend and frontend)
   Option B: Run separately:
      - start_backend.bat (Python API)
      - start_frontend.bat (React UI)

3. Open your browser:
   → http://localhost:3000

4. Integrate your trading bots:
   - Edit dashboard_api.py
   - Uncomment bot imports in start() methods
   - Add callback functions to your bots

📚 DOCUMENTATION:

See DASHBOARD_SETUP_GUIDE.md for:
- Complete integration instructions
- Troubleshooting guide
- Customization options
- Security considerations

🚀 QUICK TEST:

To verify everything works:
1. Run: python dashboard_api.py
2. In another terminal: cd frontend && npm run dev
3. Visit: http://localhost:3000
4. You should see the dashboard!

💡 TIPS:

- Backend runs on port 5000
- Frontend runs on port 3000
- Make sure TWS is running before starting bots
- Check dashboard_data/ for SQLite database
- Logs are in the Activity Log section

Happy Trading! 📈💰
""")

def main():
    """Main setup function"""
    print_header("🚀 Trading Bot Dashboard - Setup Script")
    
    print("This script will set up your trading dashboard automatically.\n")
    print("Setup includes:")
    print("  • Python package installation")
    print("  • Directory structure creation")
    print("  • Frontend configuration")
    print("  • Start scripts generation")
    print("\nPress Enter to continue or Ctrl+C to cancel...")
    input()
    
    # Step 1: Check and install Python packages
    missing_packages = check_python_packages()
    if missing_packages:
        if not install_python_packages(missing_packages):
            print("\n❌ Failed to install Python packages. Please install manually:")
            print(f"   pip install {' '.join(missing_packages)}")
            return
    
    # Step 2: Check Node.js
    if not check_node():
        print("\n❌ Node.js is required. Please install it and run this script again.")
        return
    
    # Step 3: Create directories
    create_directory_structure()
    
    # Step 4: Create frontend files
    create_frontend_files()
    
    # Step 5: Install frontend dependencies
    if not install_frontend_dependencies():
        print("\n⚠️  Frontend installation had issues. You may need to run:")
        print("   cd frontend")
        print("   npm install")
    
    # Step 6: Create start scripts
    create_start_scripts()
    
    # Done!
    print_next_steps()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled by user")
    except Exception as e:
        print(f"\n\n❌ Setup failed with error: {e}")
        print("Please check the error and try again or install manually.")

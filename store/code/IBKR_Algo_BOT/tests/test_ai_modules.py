"""
IBKR AI Trading Assistant - Installation Test Script
Run this after installation to verify everything is working correctly
"""

import sys
from pathlib import Path

# Add the bot directory to Python path so imports work
bot_dir = Path(__file__).parent.parent
sys.path.insert(0, str(bot_dir))

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def test_python_version():
    """Test Python version"""
    print("\n1. Testing Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} - Need 3.11+")
        return False

def test_imports():
    """Test if all required packages can be imported"""
    print("\n2. Testing package imports...")
    
    packages = {
        'fastapi': 'FastAPI',
        'uvicorn': 'Uvicorn',
        'pydantic': 'Pydantic',
        'ib_insync': 'IB-Insync',
        'pandas': 'Pandas',
        'numpy': 'NumPy'
    }
    
    all_ok = True
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"   ✅ {name} - OK")
        except ImportError:
            print(f"   ❌ {name} - NOT INSTALLED")
            all_ok = False
    
    return all_ok

def test_modules():
    """Test if our AI modules load correctly"""
    print("\n3. Testing AI modules...")
    
    modules_ok = True
    
    # Test market analyst
    try:
        from server.claude_integration.market_analyst import MarketAnalyst
        analyst = MarketAnalyst()
        print("   ✅ Market Analyst - OK")
    except Exception as e:
        print(f"   ❌ Market Analyst - ERROR: {e}")
        modules_ok = False
    
    # Test trade validator
    try:
        from server.claude_integration.trade_validator import TradeValidator
        validator = TradeValidator()
        print("   ✅ Trade Validator - OK")
    except Exception as e:
        print(f"   ❌ Trade Validator - ERROR: {e}")
        modules_ok = False
    
    return modules_ok

def test_api_creation():
    """Test if FastAPI app can be created"""
    print("\n4. Testing API creation...")
    
    try:
        from fastapi import FastAPI
        app = FastAPI()
        print("   ✅ FastAPI App - OK")
        return True
    except Exception as e:
        print(f"   ❌ FastAPI App - ERROR: {e}")
        return False

async def test_async_functions():
    """Test async functions"""
    print("\n5. Testing async functionality...")
    
    try:
        from server.claude_integration.market_analyst import simple_market_check
        result = await simple_market_check(['AAPL'])
        print("   ✅ Async Market Check - OK")
        return True
    except Exception as e:
        print(f"   ❌ Async Market Check - ERROR: {e}")
        return False

def test_ibkr_connection():
    """Test IBKR connection (optional)"""
    print("\n6. Testing IBKR connection (optional)...")
    
    try:
        from ib_insync import IB
        ib = IB()
        
        # Try to connect
        try:
            ib.connect('127.0.0.1', 3333, clientId=999)
            print("   ✅ IBKR Connection - CONNECTED")
            ib.disconnect()
            return True
        except:
            print("   ⚠️  IBKR Connection - NOT CONNECTED (TWS not running)")
            print("      This is OK for testing. Start TWS to enable IBKR features.")
            return True
            
    except Exception as e:
        print(f"   ❌ IBKR Connection Test - ERROR: {e}")
        return False

def print_summary(results):
    """Print test summary"""
    print_header("TEST SUMMARY")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\nTests Passed: {passed}/{total}")
    print("\nDetails:")
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} - {test_name}")
    
    if passed == total:
        print("\n" + "🎉 "*10)
        print("ALL TESTS PASSED! System is ready to use.")
        print("🎉 "*10)
        print("\nNext steps:")
        print("1. Start TWS (Trader Workstation)")
        print("2. Run: python server\\claude_integration\\claude_api.py")
        print("3. Open: http://localhost:8000/docs")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("- Install missing packages: pip install -r requirements.txt")
        print("- Make sure you're in the virtual environment")
        print("- Check Python version (need 3.11+)")

async def run_all_tests():
    """Run all tests"""
    print_header("IBKR AI TRADING ASSISTANT - INSTALLATION TEST")
    print("\nThis script will verify your installation is correct.\n")
    
    results = {}
    
    # Run tests
    results['Python Version'] = test_python_version()
    results['Package Imports'] = test_imports()
    results['AI Modules'] = test_modules()
    results['FastAPI'] = test_api_creation()
    results['Async Functions'] = await test_async_functions()
    results['IBKR Connection'] = test_ibkr_connection()
    
    # Print summary
    print_summary(results)
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    import asyncio
    
    try:
        # Run the async test suite
        asyncio.run(run_all_tests())
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Fatal error during testing: {e}")
        print("\nPlease check your installation and try again.")

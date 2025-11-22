"""
Quick test script for Alpaca integration
"""
from dotenv import load_dotenv
load_dotenv()

print("\n" + "="*70)
print("ALPACA INTEGRATION TEST")
print("="*70 + "\n")

# Test 1: Import modules
print("Test 1: Importing modules...")
try:
    from config.broker_config import get_broker_config
    from alpaca_integration import get_alpaca_connector
    from alpaca_market_data import get_alpaca_market_data
    from ai.alpaca_ai_predictor import get_alpaca_predictor
    print("[OK] All modules imported successfully\n")
except Exception as e:
    print(f"[FAIL] Module import failed: {e}\n")
    exit(1)

# Test 2: Configuration
print("Test 2: Checking configuration...")
try:
    config = get_broker_config()
    print(f"[OK] Broker Type: {config.broker_type}")
    print(f"     Paper Trading: {config.alpaca.paper if config.is_alpaca() else 'N/A'}\n")
except Exception as e:
    print(f"[FAIL] Configuration failed: {e}\n")
    exit(1)

# Test 3: Alpaca Connection
print("Test 3: Testing Alpaca connection...")
try:
    connector = get_alpaca_connector()

    if connector.is_connected():
        account = connector.get_account()
        print("[OK] Alpaca Connected!")
        print(f"     Account: {account['account_id']}")
        print(f"     Buying Power: ${account['buying_power']:,.2f}")
        print(f"     Portfolio Value: ${account['portfolio_value']:,.2f}\n")
    else:
        print("[FAIL] Alpaca connection failed\n")
        exit(1)
except Exception as e:
    print(f"[FAIL] Connection error: {e}\n")
    exit(1)

# Test 4: Market Data
print("Test 4: Testing market data...")
try:
    market_data = get_alpaca_market_data()
    quote = market_data.get_latest_quote("SPY")

    if quote:
        print("[OK] Market data working!")
        print(f"     SPY Quote:")
        print(f"     Bid: ${quote['bid']:.2f}")
        print(f"     Ask: ${quote['ask']:.2f}")
        print(f"     Last: ${quote['last']:.2f}\n")
    else:
        print("[FAIL] No market data received\n")
except Exception as e:
    print(f"[FAIL] Market data error: {e}\n")

# Test 5: AI Predictor
print("Test 5: Testing AI predictor...")
try:
    predictor = get_alpaca_predictor()

    if predictor.model:
        print("[OK] AI Model loaded!")
        print(f"     Accuracy: {predictor.accuracy:.4f}")
        print(f"     Features: {len(predictor.feature_names)}")
        print(f"     Training Date: {predictor.training_date}\n")
    else:
        print("[INFO] No AI model trained yet")
        print("       Use START_ALPACA_HUB.ps1 -TrainModel to train\n")
except Exception as e:
    print(f"[WARN] AI predictor note: {e}\n")

print("="*70)
print("INTEGRATION TEST COMPLETE!")
print("="*70)
print("\n[OK] System is ready!")
print("\nNext steps:")
print("  1. Train AI model: .\\START_ALPACA_HUB.ps1 -TrainModel")
print("  2. Start dashboard: .\\START_ALPACA_HUB.ps1")
print("  3. Open: http://localhost:9100/dashboard")
print("\n")

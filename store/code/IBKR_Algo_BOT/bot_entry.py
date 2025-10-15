import argparse
import importlib
import runpy
import sys
import traceback

# Optional router stack (for paper-demo)
try:
    from IBKR_Algo_BOT.ibkr_adapter import IBKRAdapter, IBKRConfig
    from IBKR_Algo_BOT.order_router import OrderRouter
    from IBKR_Algo_BOT.risk_manager import RiskManager, RiskConfig
    from IBKR_Algo_BOT.broker_if import Order
except Exception:
    IBKRAdapter = IBKRConfig = OrderRouter = RiskManager = RiskConfig = Order = None

def run_module(module_name: str, args=None):
    try:
        m = importlib.import_module(f"IBKR_Algo_BOT.{module_name}")
    except Exception as e:
        print(f"❌ Could not import IBKR_Algo_BOT.{module_name}: {e}")
        traceback.print_exc()
        sys.exit(2)
    fn = getattr(m, "main", None)
    if callable(fn):
        try:
            return fn() if args is None else fn(*args)
        except SystemExit as se:
            sys.exit(int(getattr(se, "code", 1) or 0))
        except Exception:
            traceback.print_exc(); sys.exit(1)
    else:
        try:
            runpy.run_module(f"IBKR_Algo_BOT.{module_name}", run_name="__main__", alter_sys=True)
            return 0
        except SystemExit as se:
            sys.exit(int(getattr(se, "code", 1) or 0))
        except Exception:
            traceback.print_exc(); sys.exit(1)

def run_paper_demo(symbol="AAPL", qty=1):
    if any(x is None for x in (IBKRAdapter, IBKRConfig, OrderRouter, RiskManager, RiskConfig, Order)):
        print("❌ Router stack not available."); sys.exit(2)
    cfg = IBKRConfig(paper=True)
    broker = IBKRAdapter(cfg); risk = RiskManager(RiskConfig()); router = OrderRouter(broker, risk)
    broker.connect(); print("✅ Connected to IBKR (paper).")
    # Example (commented):
    # oid = router.submit(Order(symbol=symbol, side="BUY", qty=qty, type="MKT"), mark_price=0.0)
    # print("Submitted OID:", oid)
    evts = router.pump_events(); print("Events:", evts); return 0

def main():
    p = argparse.ArgumentParser(prog="bot_entry", description="Unified entrypoint for IBKR_Algo_BOT")
    p.add_argument("--mode", required=True, choices=[
        "validate-ibkr","dashboard","backtest","train-lstm",
        "paper-demo","selector","backtest-cli","totalview","totalview-cli"
    ])
    p.add_argument("--symbol", default="AAPL")
    p.add_argument("--qty", type=float, default=1)
    args, unknown = p.parse_known_args()
    if args.mode == "validate-ibkr": return run_module("validate_ibkr_connection")
    if args.mode == "dashboard":     return run_module("dashboard_api")
    if args.mode == "backtest":      return run_module("improved_backtest")
    if args.mode == "train-lstm":    return run_module("lstm_training_pipeline")
    if args.mode == "selector":      return run_module("trading_strategy_selector")
    if args.mode == "paper-demo":    return run_paper_demo(args.symbol, args.qty)
    if args.mode == "backtest-cli":  return run_module("run_backtest_cli")
    if args.mode == "totalview":     return run_module("ibkr_totalview_integration")
    if args.mode == "totalview-cli": return run_module("run_totalview_cli")
    print("unknown mode"); return 2

if __name__ == "__main__":
    sys.exit(main())

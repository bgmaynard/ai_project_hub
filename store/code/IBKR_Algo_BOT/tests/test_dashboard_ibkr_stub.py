# IBKR_Algo_BOT/tests/test_dashboard_ibkr_stub.py
import pytest

@pytest.fixture
def client(monkeypatch):
    from IBKR_Algo_BOT.dashboard_api import app, ibkr_manager
    monkeypatch.setattr(ibkr_manager, "is_connected", lambda: False, raising=False)
    ibkr_manager.connected = False
    ibkr_manager.client = None
    ibkr_manager.wrapper = None
    ibkr_manager.positions = []
    ibkr_manager.orders = []
    ibkr_manager.account_value = 0
    ibkr_manager.buying_power = 0
    with app.test_client() as c:
        yield c

def test_health_ok(client):
    r = client.get("/api/health"); assert r.status_code == 200
    data = r.get_json(); assert data["status"] == "ok"; assert "timestamp" in data

def test_status_shape_when_disconnected(client):
    r = client.get("/api/status"); assert r.status_code == 200
    data = r.get_json()
    assert data["ibkr_connected"] is False
    assert data["mtf_running"] in (True, False)
    assert data["warrior_running"] in (True, False)
    assert "timestamp" in data

def test_ibkr_test_endpoint_when_disconnected(client):
    r = client.get("/api/ibkr/test"); assert r.status_code == 200
    data = r.get_json()
    assert "ibkr_available" in data
    assert data["connected"] is False
    assert data["client_exists"] in (True, False)
    assert data["wrapper_exists"] in (True, False)

def test_account_endpoint_when_disconnected(client):
    r = client.get("/api/ibkr/account"); assert r.status_code == 200
    data = r.get_json()
    assert data["connected"] is False
    assert data["account_value"] == 0
    assert data["buying_power"] == 0

def test_trade_execute_requires_connection(client):
    payload = {"symbol": "AAPL", "action": "BUY", "quantity": 1, "order_type": "MKT"}
    r = client.post("/api/trade/execute", json=payload)
    assert r.status_code == 400
    data = r.get_json()
    assert data["success"] is False
    assert "Not connected" in data["message"]

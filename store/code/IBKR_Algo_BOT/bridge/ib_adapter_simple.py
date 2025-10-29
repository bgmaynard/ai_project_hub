"""
IBKR Adapter - Exact copy of working test code
"""
import os
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from dotenv import load_dotenv

# Load environment
project_root = Path(__file__).parent.parent.parent.parent
env_file = project_root / '.env'
load_dotenv(env_file)

from ib_insync import IB

class IBConfig:
    def __init__(self):
        self.host = os.getenv("TWS_HOST", "127.0.0.1")
        self.port = int(os.getenv("TWS_PORT", "7497"))
        self.base_client_id = int(os.getenv("TWS_CLIENT_ID", "1"))
        self.connect_timeout = 30.0  # Hardcoded to match working test
        print(f"IBConfig: {self.host}:{self.port}, clientId: {self.base_client_id}, timeout: {self.connect_timeout}s")

class IBConnectionState:
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    FAILED = "FAILED"

class IBAdapter:
    """Dead simple adapter - exact copy of working test"""
    
    def __init__(self, config: IBConfig):
        self.config = config
        self.ib = None
        self.current_client_id = config.base_client_id
        self.connection_state = IBConnectionState.DISCONNECTED
        self.last_error = None
        
    def connect(self) -> bool:
        """Connect using EXACT code from working test"""
        print(f"Connecting to TWS on {self.config.port}...")
        self.connection_state = IBConnectionState.CONNECTING
        
        try:
            # EXACT code from working test
            self.ib = IB()
            self.ib.connect(self.config.host, self.config.port, clientId=self.current_client_id, timeout=30)
            
            # Success!
            self.connection_state = IBConnectionState.CONNECTED
            print("SUCCESS! Connected!")
            print(f"Server version: {self.ib.client.serverVersion()}")
            return True
            
        except Exception as e:
            self.last_error = str(e)
            self.connection_state = IBConnectionState.FAILED
            print(f"Failed: {e}")
            return False
    
    def disconnect(self):
        if self.ib and self.ib.isConnected():
            try:
                self.ib.disconnect()
                self.connection_state = IBConnectionState.DISCONNECTED
                print("Disconnected")
            except Exception as e:
                print(f"Disconnect error: {e}")
    
    def is_connected(self) -> bool:
        return (self.connection_state == IBConnectionState.CONNECTED and 
                self.ib and self.ib.isConnected())
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "state": self.connection_state,
            "host": self.config.host,
            "port": self.config.port,
            "current_client_id": self.current_client_id,
            "last_error": self.last_error,
            "ib_connected": self.ib.isConnected() if self.ib else False,
            "timestamp": datetime.utcnow().isoformat()
        }

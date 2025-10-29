"""
Enhanced IBKR Adapter - Using synchronous connection (which works!)
"""
import os
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import socket
from dotenv import load_dotenv

# Load environment
project_root = Path(__file__).parent.parent.parent.parent
env_file = project_root / '.env'
load_dotenv(env_file)

# Import ib_insync
try:
    from ib_insync import IB
except ImportError as e:
    print(f"ERROR: ib_insync not installed: {e}")
    raise

class IBConfig:
    """IBKR Configuration from environment"""
    
    def __init__(self):
        self.host = os.getenv("TWS_HOST", "127.0.0.1")
        self.port = int(os.getenv("TWS_PORT", "7497"))
        self.base_client_id = int(os.getenv("TWS_CLIENT_ID", "1"))
        self.connect_timeout = float(os.getenv("IB_CONNECT_TIMEOUT_SEC", "30.0"))
        
        print(f"IBConfig: {self.host}:{self.port}, clientId: {self.base_client_id}, timeout: {self.connect_timeout}s")
        
    def test_connection(self) -> bool:
        """Test if TWS is listening"""
        try:
            with socket.create_connection((self.host, self.port), timeout=3) as sock:
                print(f"SUCCESS: TWS is listening on {self.host}:{self.port}")
                return True
        except Exception as e:
            print(f"ERROR: TWS not listening: {e}")
            return False

class IBConnectionState:
    """Connection states"""
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    FAILED = "FAILED"

class IBAdapter:
    """IBKR Adapter using synchronous connection (which works!)"""
    
    def __init__(self, config: IBConfig):
        self.config = config
        self.ib = IB()
        self.current_client_id = config.base_client_id
        self.connection_state = IBConnectionState.DISCONNECTED
        self.logger = logging.getLogger(__name__)
        self.last_error = None
        
    def connect(self) -> bool:
        """
        Connect to IBKR using SYNCHRONOUS method (which works!)
        Note: This is intentionally NOT async!
        """
        
        print(f"Connecting to {self.config.host}:{self.config.port}")
        self.connection_state = IBConnectionState.CONNECTING
        
        # Test if TWS is listening first
        if not self.config.test_connection():
            self.connection_state = IBConnectionState.FAILED
            self.last_error = "TWS not listening"
            return False
        
        # Try multiple client IDs
        for attempt in range(5):
            client_id = self.config.base_client_id + attempt
            
            try:
                print(f"Attempt {attempt + 1}/5 with clientId: {client_id}")
                
                # Use SYNCHRONOUS connect (not connectAsync!)
                self.ib.connect(
                    host=self.config.host,
                    port=self.config.port,
                    clientId=client_id,
                    timeout=self.config.connect_timeout
                )
                
                # Success!
                self.current_client_id = client_id
                self.connection_state = IBConnectionState.CONNECTED
                print(f"SUCCESS: Connected with clientId: {client_id}")
                
                # Test basic functionality
                try:
                    current_time = self.ib.reqCurrentTime()
                    print(f"SUCCESS: TWS time: {current_time}")
                except Exception as e:
                    print(f"WARNING: Time test failed: {e}")
                
                return True
                
            except Exception as e:
                error_msg = str(e)
                self.last_error = error_msg
                print(f"ERROR: {error_msg}")
                
                # Handle specific errors
                if "326" in error_msg or "in use" in error_msg.lower():
                    print(f"ClientId {client_id} in use, trying next...")
                    continue
                elif "timeout" in error_msg.lower():
                    print(f"TIMEOUT: clientId {client_id}")
                    continue
                else:
                    break
        
        # All attempts failed
        self.connection_state = IBConnectionState.FAILED
        print("ERROR: All connection attempts failed")
        return False
    
    def disconnect(self):
        """Disconnect from TWS"""
        if self.ib and self.ib.isConnected():
            try:
                self.ib.disconnect()
                self.connection_state = IBConnectionState.DISCONNECTED
                print("SUCCESS: Disconnected")
            except Exception as e:
                print(f"ERROR: Disconnect error: {e}")
    
    def is_connected(self) -> bool:
        """Check if connected"""
        return (self.connection_state == IBConnectionState.CONNECTED and 
                self.ib and self.ib.isConnected())
    
    def get_status(self) -> Dict[str, Any]:
        """Get detailed status"""
        return {
            "state": self.connection_state,
            "host": self.config.host,
            "port": self.config.port,
            "current_client_id": self.current_client_id,
            "last_error": self.last_error,
            "ib_connected": self.ib.isConnected() if self.ib else False,
            "timestamp": datetime.utcnow().isoformat()
        }

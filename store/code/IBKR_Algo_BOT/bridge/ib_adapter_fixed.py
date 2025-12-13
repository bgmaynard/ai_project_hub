"""
Enhanced IBKR Adapter with Auto-Reconnect and Diagnostics
"""
import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import socket

# Load environment first
from dotenv import load_dotenv
project_root = Path(__file__).parent.parent.parent.parent
env_file = project_root / '.env'
load_dotenv(env_file)

# Import ib_insync
try:
    from ib_insync import IB
except ImportError as e:
    print(f"ERROR: ib_insync not installed: {e}")
    print("SOLUTION: Run pip install ib_insync")
    raise

class IBConfig:
    """IBKR Configuration from environment"""
    
    def __init__(self):
        self.host = os.getenv("TWS_HOST", "127.0.0.1")
        self.port = int(os.getenv("TWS_PORT", "7497"))
        self.base_client_id = int(os.getenv("TWS_CLIENT_ID", "6001"))
        self.heartbeat_interval = float(os.getenv("IB_HEARTBEAT_SEC", "3.0"))
        self.connect_timeout = float(os.getenv("IB_CONNECT_TIMEOUT_SEC", "60.0"))
        
        print(f"IBConfig: {self.host}:{self.port}, clientId: {self.base_client_id}, timeout: {self.connect_timeout}s")
        
    def test_connection(self) -> bool:
        """Test if TWS is listening"""
        try:
            with socket.create_connection((self.host, self.port), timeout=3) as sock:
                print(f"SUCCESS: TWS is listening on {self.host}:{self.port}")
                return True
        except ConnectionRefusedError:
            print(f"ERROR: TWS not listening on {self.host}:{self.port}")
            print("SOLUTION: Start TWS and enable API in settings")
            return False
        except Exception as e:
            print(f"WARNING: Connection test failed: {e}")
            return False

class IBConnectionState:
    """Connection states"""
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    DEGRADED = "DEGRADED"
    RECONNECTING = "RECONNECTING"
    FAILED = "FAILED"

class IBAdapter:
    """Enhanced IBKR Adapter with diagnostics"""
    
    def __init__(self, config: IBConfig):
        self.config = config
        self.ib = None
        self.current_client_id = config.base_client_id
        self.connection_state = IBConnectionState.DISCONNECTED
        self.logger = logging.getLogger(__name__)
        self.last_error = None
        
    async def connect(self) -> bool:
        """Connect to IBKR with retry logic"""
        
        self.logger.info(f"Connecting to {self.config.host}:{self.config.port} with timeout {self.config.connect_timeout}s")
        self.connection_state = IBConnectionState.CONNECTING
        
        # Test if TWS is listening first
        if not self.config.test_connection():
            self.connection_state = IBConnectionState.FAILED
            self.last_error = "TWS not listening"
            return False
        
        # Try multiple client IDs
        for attempt in range(10):
            client_id = self.config.base_client_id + attempt
            
            try:
                self.logger.info(f"Attempt {attempt + 1}/10 with clientId: {client_id}, timeout: {self.config.connect_timeout}s")
                
                self.ib = IB()
                
                # CRITICAL: Use the timeout from config
                await self.ib.connectAsync(
                    host=self.config.host,
                    port=self.config.port,
                    clientId=client_id,
                    timeout=self.config.connect_timeout
                )
                
                # Success!
                self.current_client_id = client_id
                self.connection_state = IBConnectionState.CONNECTED
                self.logger.info(f"SUCCESS: Connected with clientId: {client_id}")
                
                # Test basic functionality
                try:
                    current_time = await self.ib.reqCurrentTimeAsync()
                    self.logger.info(f"SUCCESS: TWS time: {current_time}")
                except Exception as e:
                    self.logger.warning(f"WARNING: Time test failed: {e}")
                    self.connection_state = IBConnectionState.DEGRADED
                
                return True
                
            except asyncio.TimeoutError:
                self.logger.warning(f"TIMEOUT: clientId {client_id} after {self.config.connect_timeout}s")
                continue
                
            except Exception as e:
                error_msg = str(e)
                self.last_error = error_msg
                self.logger.warning(f"ERROR: {error_msg}")
                
                # Handle specific errors
                if "326" in error_msg:
                    self.logger.info(f"ClientId {client_id} in use, trying next...")
                    continue
                elif "502" in error_msg:
                    self.logger.error("ERROR: TWS not ready for API")
                    break
                elif "refused" in error_msg.lower():
                    self.logger.error("ERROR: Connection refused")
                    break
                else:
                    continue
        
        # All attempts failed
        self.connection_state = IBConnectionState.FAILED
        self.logger.error("ERROR: All connection attempts failed")
        return False
    
    async def disconnect(self):
        """Disconnect from TWS"""
        if self.ib and self.ib.isConnected():
            try:
                self.ib.disconnect()
                self.connection_state = IBConnectionState.DISCONNECTED
                self.logger.info("SUCCESS: Disconnected")
            except Exception as e:
                self.logger.error(f"ERROR: Disconnect error: {e}")
    
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

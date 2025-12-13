"""
Unified Broker Configuration Manager
Supports both Alpaca and IBKR brokers
"""
import os
from enum import Enum
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


class BrokerType(str, Enum):
    """Supported broker types"""
    ALPACA = "alpaca"
    IBKR = "ibkr"


class AlpacaConfig(BaseModel):
    """Alpaca API configuration"""
    api_key: str
    secret_key: str
    paper: bool = True
    base_url: Optional[str] = "https://paper-api.alpaca.markets"


class IBKRConfig(BaseModel):
    """IBKR API configuration"""
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 1
    readonly: bool = False


class BrokerConfig:
    """Main broker configuration manager"""

    def __init__(self, broker_type: Optional[BrokerType] = None):
        """
        Initialize broker configuration

        Args:
            broker_type: Type of broker to use (defaults to ALPACA)
        """
        # Determine broker type from environment or use default
        self.broker_type = broker_type or BrokerType(
            os.getenv('BROKER_TYPE', 'alpaca').lower()
        )

        # Load appropriate configuration
        if self.broker_type == BrokerType.ALPACA:
            self.alpaca = self._load_alpaca_config()
            self.ibkr = None
        elif self.broker_type == BrokerType.IBKR:
            self.ibkr = self._load_ibkr_config()
            self.alpaca = None
        else:
            raise ValueError(f"Unsupported broker type: {self.broker_type}")

    def _load_alpaca_config(self) -> AlpacaConfig:
        """Load Alpaca configuration from environment"""
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')

        if not api_key or not secret_key:
            raise ValueError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env file"
            )

        return AlpacaConfig(
            api_key=api_key,
            secret_key=secret_key,
            paper=os.getenv('ALPACA_PAPER', 'true').lower() == 'true',
            base_url=os.getenv('ALPACA_ENDPOINT', 'https://paper-api.alpaca.markets')
        )

    def _load_ibkr_config(self) -> IBKRConfig:
        """Load IBKR configuration from environment"""
        return IBKRConfig(
            host=os.getenv('IBKR_HOST', '127.0.0.1'),
            port=int(os.getenv('IBKR_PORT', '7497')),
            client_id=int(os.getenv('IBKR_CLIENT_ID', '1')),
            readonly=os.getenv('IBKR_READONLY', 'false').lower() == 'true'
        )

    def get_config_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary"""
        if self.broker_type == BrokerType.ALPACA:
            return {
                "broker": "alpaca",
                "paper_trading": self.alpaca.paper,
                "api_key": f"{self.alpaca.api_key[:8]}...",  # Masked
                "endpoint": self.alpaca.base_url
            }
        else:
            return {
                "broker": "ibkr",
                "host": self.ibkr.host,
                "port": self.ibkr.port,
                "client_id": self.ibkr.client_id,
                "readonly": self.ibkr.readonly
            }

    def is_alpaca(self) -> bool:
        """Check if using Alpaca"""
        return self.broker_type == BrokerType.ALPACA

    def is_ibkr(self) -> bool:
        """Check if using IBKR"""
        return self.broker_type == BrokerType.IBKR


# Global configuration instance
_config_instance: Optional[BrokerConfig] = None


def get_broker_config(broker_type: Optional[BrokerType] = None) -> BrokerConfig:
    """
    Get or create the global broker configuration instance

    Args:
        broker_type: Type of broker to use (only used on first call)

    Returns:
        BrokerConfig instance
    """
    global _config_instance

    if _config_instance is None:
        _config_instance = BrokerConfig(broker_type)

    return _config_instance


def reset_broker_config():
    """Reset the global configuration (useful for testing)"""
    global _config_instance
    _config_instance = None

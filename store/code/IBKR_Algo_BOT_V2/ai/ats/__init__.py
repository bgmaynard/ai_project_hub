"""
ATS (Advanced Trading Signal) Feed System

SmartZone pattern detection with expansion breakout triggers.
Integrates with Gating Engine and HFT Scalper for trade approval.

Key Components:
- types.py: Shared types (Bar, SmartZoneSignal, MarketContext, AtsTrigger)
- time_utils.py: Time utilities, post-open guard
- smartzone.py: SmartZone pattern detector
- ats_detector.py: ATS scoring + state machine
- ats_registry.py: Per-symbol state tracking
- ats_feed.py: Main feed integration
- gating_hook.py: Gating engine integration
- scalper_hook.py: Scalper strategy hook
"""

from .types import Bar, SmartZoneSignal, MarketContext, AtsTrigger, AtsState
from .ats_detector import AtsDetector, get_ats_detector
from .ats_registry import AtsRegistry, get_ats_registry
from .ats_feed import AtsFeed, get_ats_feed
from .gating_hook import AtsGatingHook, get_ats_gating_hook
from .scalper_hook import AtsScalperHook, get_ats_scalper_hook
from .schwab_adapter import wire_schwab_to_ats, unwire_schwab_from_ats, is_wired

__all__ = [
    'Bar', 'SmartZoneSignal', 'MarketContext', 'AtsTrigger', 'AtsState',
    'AtsDetector', 'get_ats_detector',
    'AtsRegistry', 'get_ats_registry',
    'AtsFeed', 'get_ats_feed',
    'AtsGatingHook', 'get_ats_gating_hook',
    'AtsScalperHook', 'get_ats_scalper_hook',
    'wire_schwab_to_ats', 'unwire_schwab_from_ats', 'is_wired',
]

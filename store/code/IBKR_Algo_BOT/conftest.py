import os
from pathlib import Path

INTEG = os.getenv('RUN_INTEGRATION') == '1'
INTEGRATION_DIR_NAME = 'tests_integration'

def pytest_ignore_collect(collection_path=None, path=None, config=None):
    # Support pytest <9 (path) and >=9 (collection_path)
    p = None
    if collection_path is not None:
        p = Path(collection_path)
    elif path is not None:
        p = Path(str(path))

    if not p:
        return False

    # Ignore the integration folder unless explicitly enabled
    if INTEGRATION_DIR_NAME in p.parts and not INTEG:
        return True

    # Ignore known heavy test files by name if they sneak in
    heavy = {'test_lstm.py','test_ibkr.py','totalview_test.py','totalview_live_test.py','level1_test.py'}
    if p.name in heavy and not INTEG:
        return True

    return False
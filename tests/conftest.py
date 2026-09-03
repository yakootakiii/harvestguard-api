"""Put the repository root on sys.path.

`python -m pytest` adds the CWD automatically, but bare `pytest` does not, so
without this `from app.model import ...` resolves only when invoked one specific
way.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

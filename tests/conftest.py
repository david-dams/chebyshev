import sys
from pathlib import Path

# The project is a collection of standalone scripts (no installed package),
# so make `scripts/` importable for the tests.
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

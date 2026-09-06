import os
import sys
from pathlib import Path

os.environ.setdefault("ALLOWED_ORIGINS", "http://localhost")
os.environ.setdefault("HOPSWORKS_API_KEY", "test-key")

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "src" / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

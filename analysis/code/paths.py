"""Repository root, resolved relative to this file so the scripts run from any checkout location."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2].as_posix() + "/"

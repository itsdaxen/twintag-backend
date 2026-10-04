"""Vercel ASGI entrypoint for the hosted TwinTag demo API."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twintag_backend.main import app

__all__ = ["app"]

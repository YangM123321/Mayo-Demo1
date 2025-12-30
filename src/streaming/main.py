from __future__ import annotations

from src.streaming.consumer import run_forever
from src.streaming.handler import handle_vital

if __name__ == "__main__":
    run_forever(handle_vital)

# src/streaming/main.py
from src.streaming.consumer import run_forever

def handler(evt):
    # minimal “processing” for CI: write to audit log
    with open("/tmp/vitals_audit.log", "a", encoding="utf-8") as f:
        f.write(evt.model_dump_json() + "\n")

if __name__ == "__main__":
    run_forever(handler)

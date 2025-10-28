# streaming/consumer_to_features.py
import os, json, time
from pathlib import Path
from datetime import datetime
import pandas as pd
from confluent_kafka import Consumer

BROKER  = os.getenv("KAFKA_BROKER", "localhost:19092")
TOPIC   = os.getenv("KAFKA_TOPIC", "vitals")
OUT_DIR = Path("data/stream_features"); OUT_DIR.mkdir(parents=True, exist_ok=True)

c = Consumer({
    "bootstrap.servers": BROKER,
    "group.id": "feat-agg",
    "enable.auto.commit": True,
    "auto.offset.reset": "earliest"
})
c.subscribe([TOPIC])

buf = []
FLUSH_SEC = 10
last = time.time()

def flush():
    global buf
    if not buf: return
    df = pd.DataFrame(buf); buf = []
    agg = (df.groupby("patient_id")
            .agg(BP_SYS_mean=("BP_SYS","mean"),
                 BP_SYS_last=("BP_SYS","last"),
                 BP_DIA_mean=("BP_DIA","mean"))
            .reset_index())
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    out = OUT_DIR / f"features_{ts}.parquet"
    agg.to_parquet(out, index=False)
    print(f"Wrote {len(agg)} rows → {out}")

try:
    while True:
        m = c.poll(1.0)
        if m is None: pass
        elif m.error(): print("❌", m.error())
        else: buf.append(json.loads(m.value().decode("utf-8")))
        if time.time() - last >= FLUSH_SEC:
            flush(); last = time.time()
except KeyboardInterrupt:
    pass
finally:
    flush()
    c.close()

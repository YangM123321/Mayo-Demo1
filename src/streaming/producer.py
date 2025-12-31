import os, json, time, argparse
from datetime import datetime
from confluent_kafka import Producer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--topic", default=os.getenv("KAFKA_TOPIC_IN", "vitals.in"))
    args = ap.parse_args()

    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "127.0.0.1:9092")
    print(f"[producer] bootstrap={bootstrap} topic={args.topic} n={args.n}", flush=True)

    p = Producer({
        "bootstrap.servers": bootstrap,
        # fail fast (no hanging)
        "socket.timeout.ms": 5000,
        "request.timeout.ms": 5000,
        "message.timeout.ms": 5000,
        "retries": 2,
        "retry.backoff.ms": 250,
    })

    # IMPORTANT: callbacks need poll() to be served
    delivered = 0
    def on_delivery(err, msg):
        nonlocal delivered
        if err is not None:
            raise RuntimeError(f"Delivery failed: {err}")
        delivered += 1

    for i in range(args.n):
        event = {
            "event_id": f"smoke-{int(time.time()*1000)}-{i}",
            "ts": datetime.utcnow().isoformat() + "Z",
        }
        p.produce(args.topic, value=json.dumps(event).encode("utf-8"), on_delivery=on_delivery)
        p.poll(0)

    remaining = p.flush(10)   # CRITICAL: timeout so it can't hang forever
    if remaining != 0:
        raise RuntimeError(f"[producer] flush timed out, remaining={remaining}")

    print(f"[producer] ✅ produced={args.n} delivered={delivered}", flush=True)

if __name__ == "__main__":
    main()

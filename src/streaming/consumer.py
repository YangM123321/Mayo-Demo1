from __future__ import annotations

import json
import os
import signal
import sys
import time
from datetime import datetime
from typing import Any, Optional

from confluent_kafka import Consumer, KafkaException


def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v else default


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def _write_audit_line(audit_path: str, record: dict[str, Any]) -> None:
    _ensure_parent_dir(audit_path)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    # Use line-buffered writes so CI sees output quickly
    with open(audit_path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()


class _StopFlag:
    stop: bool = False


def main() -> int:
    bootstrap = _env("KAFKA_BOOTSTRAP_SERVERS", "127.0.0.1:9092")
    topic_in = _env("KAFKA_TOPIC_IN", "vitals.in")
    group_id = _env("KAFKA_GROUP_ID", "ci-consumer-local-0")
    audit_path = _env("VITALS_AUDIT_PATH", "/tmp/vitals_audit.log")

    # Optional knobs
    auto_offset_reset = _env("KAFKA_AUTO_OFFSET_RESET", "earliest")  # earliest|latest
    poll_timeout_s = float(_env("KAFKA_POLL_TIMEOUT_S", "0.5"))

    # IMPORTANT:
    # - For CI smoke test, we want the consumer to NOT exit early.
    # - We keep polling until killed by the pwsh script.
    cfg: dict[str, Any] = {
        "bootstrap.servers": bootstrap,
        "group.id": group_id,
        "enable.auto.commit": True,
        "auto.offset.reset": auto_offset_reset,
        # be resilient in small CI boxes
        "session.timeout.ms": 10000,
        "max.poll.interval.ms": 300000,
    }

    c = Consumer(cfg)
    c.subscribe([topic_in])

    stop = _StopFlag()

    def _handle_sigterm(signum: int, frame: Optional[object]) -> None:
        stop.stop = True

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    # Touch the audit file path exists (but still empty until we consume)
    _ensure_parent_dir(audit_path)

    try:
        while not stop.stop:
            msg = c.poll(poll_timeout_s)

            if msg is None:
                # no message yet; keep waiting
                continue

            if msg.error():
                # Non-fatal errors can happen during rebalance; log & continue
                _write_audit_line(
                    audit_path,
                    {
                        "ts": _now_iso(),
                        "type": "kafka_error",
                        "error": str(msg.error()),
                    },
                )
                continue

            try:
                raw = msg.value()
                # raw may be bytes
                if raw is None:
                    payload: Any = None
                else:
                    s = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
                    try:
                        payload = json.loads(s)
                    except Exception:
                        payload = s  # keep as string if not JSON

                record = {
                    "ts": _now_iso(),
                    "topic": msg.topic(),
                    "partition": msg.partition(),
                    "offset": msg.offset(),
                    "key": (msg.key().decode("utf-8") if msg.key() else None),
                    "payload": payload,
                }
                _write_audit_line(audit_path, record)

            except Exception as e:
                # Never crash the consumer in CI; write error and keep polling
                _write_audit_line(
                    audit_path,
                    {
                        "ts": _now_iso(),
                        "type": "consumer_exception",
                        "error": repr(e),
                    },
                )

        return 0

    except KafkaException as e:
        _write_audit_line(
            audit_path,
            {"ts": _now_iso(), "type": "kafka_exception", "error": repr(e)},
        )
        return 2

    finally:
        try:
            c.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())

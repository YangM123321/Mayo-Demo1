# beam/stream_to_bq.py
import argparse
import json
import datetime as dt
from datetime import timezone

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, StandardOptions, SetupOptions
from apache_beam.transforms.window import FixedWindows
from apache_beam.transforms.trigger import AfterWatermark, AccumulationMode


# -------- Pub/Sub -> Python tuple helpers --------
class ParseJSON(beam.DoFn):
    """Parses Pub/Sub bytes -> dict. Expects keys: patient_id, BP_SYS, BP_DIA, ts (ISO8601)."""
    def process(self, msg: bytes):
        d = json.loads(msg.decode("utf-8"))
        if "patient_id" not in d or "ts" not in d:
            return
        yield d


class WithEventTime(beam.DoFn):
    """Attach event time from the 'ts' field (ISO 8601 string)."""
    def process(self, d):
        ts = d["ts"]
        if ts.endswith("Z"):
            event_dt = dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
        else:
            event_dt = dt.datetime.fromisoformat(ts)
        event_ts = event_dt.timestamp()
        # downstream combine expects (patient_id, dict_of_values, event_ts)
        yield beam.window.TimestampedValue((int(d["patient_id"]), d, event_ts), event_ts)


# -------- Aggregation --------
class VitalAggFn(beam.CombineFn):
    """Per-key aggregator for fixed windows."""
    def create_accumulator(self):
        return {
            "sys_sum": 0.0, "sys_count": 0,
            "dia_sum": 0.0, "dia_count": 0,
            "last_sys_ts": None, "last_sys_val": None,
        }

    def add_input(self, acc, element):
        _pid, d, ev_ts = element
        sysv = d.get("BP_SYS")
        diav = d.get("BP_DIA")

        if sysv is not None:
            acc["sys_sum"] += float(sysv)
            acc["sys_count"] += 1
            if acc["last_sys_ts"] is None or ev_ts >= acc["last_sys_ts"]:
                acc["last_sys_ts"] = ev_ts
                acc["last_sys_val"] = float(sysv)

        if diav is not None:
            acc["dia_sum"] += float(diav)
            acc["dia_count"] += 1

        return acc

    def merge_accumulators(self, accs):
        out = self.create_accumulator()
        for a in accs:
            out["sys_sum"] += a["sys_sum"]
            out["sys_count"] += a["sys_count"]
            out["dia_sum"] += a["dia_sum"]
            out["dia_count"] += a["dia_count"]
            if a["last_sys_ts"] is not None and (
                out["last_sys_ts"] is None or a["last_sys_ts"] >= out["last_sys_ts"]
            ):
                out["last_sys_ts"] = a["last_sys_ts"]
                out["last_sys_val"] = a["last_sys_val"]
        return out

    def extract_output(self, acc):
        sys_mean = (acc["sys_sum"] / acc["sys_count"]) if acc["sys_count"] else None
        dia_mean = (acc["dia_sum"] / acc["dia_count"]) if acc["dia_count"] else None
        return {
            "BP_SYS_mean": sys_mean,
            "BP_SYS_last": acc["last_sys_val"],
            "BP_DIA_mean": dia_mean,
        }


class FormatForBQ(beam.DoFn):
    """Format ((patient_id), agg) -> BigQuery row dict."""
    def process(self, kv, window=beam.DoFn.WindowParam):
        patient_id, agg = kv
        if not any([agg.get("BP_SYS_mean"), agg.get("BP_SYS_last"), agg.get("BP_DIA_mean")]):
            return

        window_end = window.end.to_utc_datetime().astimezone(timezone.utc).replace(tzinfo=None)

        yield {
            "patient_id": int(patient_id),
            "BP_SYS_mean": float(agg["BP_SYS_mean"]) if agg["BP_SYS_mean"] is not None else None,
            "BP_SYS_last": float(agg["BP_SYS_last"]) if agg["BP_SYS_last"] is not None else None,
            "BP_DIA_mean": float(agg["BP_DIA_mean"]) if agg["BP_DIA_mean"] is not None else None,
            "window_end_ts": window_end.isoformat(sep=" "),
        }


# -------- Main --------
def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--region", default="us-central1")
    parser.add_argument("--input_topic", required=True, help="projects/PROJECT/topics/TOPIC")
    parser.add_argument("--bq_dataset", default="mayo")
    parser.add_argument("--bq_table", default="features_latest")
    parser.add_argument("--temp_location", required=True)
    parser.add_argument("--staging_location", required=True)
    parser.add_argument("--service_account_email", default=None)
    parser.add_argument("--window_sec", type=int, default=10)
    args, beam_args = parser.parse_known_args()

    table_spec = f"{args.project}:{args.bq_dataset}.{args.bq_table}"
    print("TableSpec =", table_spec, "Topic =", args.input_topic, flush=True)

    # Build Beam flags list and pass ONLY that list to PipelineOptions
    beam_args += [
        f"--project={args.project}",
        f"--region={args.region}",
        f"--temp_location={args.temp_location}",
        f"--staging_location={args.staging_location}",
        f"--job_name=pubsub-to-bq-agg-{dt.datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
        "--streaming",
        "--save_main_session",
        "--experiments=use_beam_bq_sink",
        "--allow_unsafe_triggers",  # required for GroupByKey + on-time-only trigger
    ]
    if args.service_account_email:
        beam_args += [f"--service_account_email={args.service_account_email}"]

    pipeline_options = PipelineOptions(beam_args)
    pipeline_options.view_as(StandardOptions).runner = "DataflowRunner"
    pipeline_options.view_as(SetupOptions).save_main_session = True

    schema = {
        "fields": [
            {"name": "patient_id", "type": "INTEGER"},
            {"name": "BP_SYS_mean", "type": "FLOAT"},
            {"name": "BP_SYS_last", "type": "FLOAT"},
            {"name": "BP_DIA_mean", "type": "FLOAT"},
            {"name": "window_end_ts", "type": "TIMESTAMP"},
        ]
    }

    with beam.Pipeline(options=pipeline_options) as p:
        (
            p
            | "ReadPubSub" >> beam.io.ReadFromPubSub(topic=args.input_topic)
            | "ParseJSON" >> beam.ParDo(ParseJSON())
            | "AttachEventTime" >> beam.ParDo(WithEventTime())
            | "KeyByPatient" >> beam.Map(lambda tup: (tup[0], tup))  # key = patient_id
            | "Window10s" >> beam.WindowInto(
                FixedWindows(args.window_sec),
                allowed_lateness=60 * 60 * 24,   # consider late data up to 24h
                trigger=AfterWatermark(),        # single on-time firing (no early/late)
                accumulation_mode=AccumulationMode.DISCARDING,
            )
            | "CombineVitals" >> beam.CombinePerKey(VitalAggFn())
            | "FormatForBQ" >> beam.ParDo(FormatForBQ())
            | "WriteToBQ" >> beam.io.gcp.bigquery.WriteToBigQuery(
                table=table_spec,
                schema=schema,
                write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
                create_disposition=beam.io.BigQueryDisposition.CREATE_NEVER,
            )
        )


if __name__ == "__main__":
    run()

# tools/dataflow_vitals_to_bq.py
import argparse, json
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, StandardOptions
from apache_beam.transforms.window import FixedWindows
from datetime import datetime, timezone

def parse_json(msg_bytes):
    rec = json.loads(msg_bytes.decode("utf-8"))
    # enforce types
    return {
        "patient_id": int(rec["patient_id"]),
        "BP_SYS": int(rec["BP_SYS"]),
        "BP_DIA": int(rec["BP_DIA"]),
        "ts": rec.get("ts"),
    }

class MakeBQRow(beam.DoFn):
    def process(self, element, window=beam.DoFn.WindowParam):
        # element is (patient_id, iterable_of_dicts)
        pid, rows = element
        rows = list(rows)
        if not rows:
            return
        sys_vals = [r["BP_SYS"] for r in rows]
        dia_vals = [r["BP_DIA"] for r in rows]
        # window end in UTC
        wend = datetime.fromtimestamp(window.end.micros / 1e6, tz=timezone.utc)
        yield {
            "patient_id": pid,
            "BP_SYS_mean": sum(sys_vals) / len(sys_vals),
            "BP_SYS_last": sys_vals[-1],
            "BP_DIA_mean": sum(dia_vals) / len(dia_vals),
            "window_end_ts": wend.isoformat(),
        }

def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_topic", required=True)
    ap.add_argument("--output_table", required=True)  # project:dataset.table
    ap.add_argument("--window_sec", type=int, default=10)
    args, beam_args = ap.parse_known_args()

    opts = PipelineOptions(beam_args, save_main_session=True, streaming=True)
    opts.view_as(StandardOptions).streaming = True

    schema = (
        "patient_id:INTEGER,"
        "BP_SYS_mean:FLOAT,"
        "BP_SYS_last:FLOAT,"
        "BP_DIA_mean:FLOAT,"
        "window_end_ts:TIMESTAMP"
    )

    with beam.Pipeline(options=opts) as p:
        (
            p
            | "ReadPubSub" >> beam.io.ReadFromPubSub(topic=args.input_topic)
            | "ParseJSON"  >> beam.Map(parse_json)
            | "KeyByPID"   >> beam.Map(lambda r: (r["patient_id"], r))
            | "Window"     >> beam.WindowInto(FixedWindows(args.window_sec))
            | "Group"      >> beam.GroupByKey()
            | "ToBQRow"    >> beam.ParDo(MakeBQRow())
            | "WriteBQ"    >> beam.io.WriteToBigQuery(
                table=args.output_table,
                schema=schema,
                write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
                create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
                custom_gcs_temp_location=None,
            )
        )

if __name__ == "__main__":
    run()

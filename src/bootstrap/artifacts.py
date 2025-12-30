from __future__ import annotations

import hashlib
import json
import os
import pathlib
import tempfile
from dataclasses import dataclass
from typing import Optional, Tuple

import yaml
from google.cloud import storage

# Cloud Run writable area. (Also works locally.)
ARTIFACT_ROOT = pathlib.Path(os.getenv("ARTIFACT_DIR", "/tmp/artifacts")).resolve()


def _parse_gcs_uri(uri: str) -> Tuple[str, str]:
    """
    Parse gs://bucket/path/to/blob into (bucket, blob).
    """
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs://... uri, got: {uri}")
    path = uri[5:]
    parts = path.split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Invalid GCS URI: {uri}")
    return parts[0], parts[1]


def _sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class DownloadSpec:
    uri: str
    dest_rel: str  # relative to ARTIFACT_ROOT
    min_bytes: int = 1
    expected_sha256: Optional[str] = None  # if provided, must match


class ArtifactError(RuntimeError):
    pass


def download_from_gcs(spec: DownloadSpec, *, force: bool = False) -> pathlib.Path:
    """
    Downloads a GCS object to ARTIFACT_ROOT/spec.dest_rel atomically.
    Validates:
      - size >= min_bytes
      - sha256 matches expected_sha256 (optional)
    """
    bucket_name, blob_name = _parse_gcs_uri(spec.uri)
    dest_path = (ARTIFACT_ROOT / spec.dest_rel).resolve()

    # Prevent path escape
    if ARTIFACT_ROOT not in dest_path.parents and dest_path != ARTIFACT_ROOT:
        raise ArtifactError(f"Refusing to write outside ARTIFACT_ROOT: {dest_path}")

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # If exists and not force, validate quickly
    if dest_path.exists() and not force:
        size = dest_path.stat().st_size
        if size < spec.min_bytes:
            raise ArtifactError(f"Existing file too small: {dest_path} ({size} bytes)")
        if spec.expected_sha256:
            actual = _sha256_file(dest_path)
            if actual.lower() != spec.expected_sha256.lower():
                raise ArtifactError(
                    f"Existing file sha mismatch for {dest_path}: {actual} != {spec.expected_sha256}"
                )
        return dest_path

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # Fetch remote size first if available
    blob.reload()  # loads size, etag, md5_hash, etc.
    remote_size = blob.size or 0
    if remote_size < spec.min_bytes:
        raise ArtifactError(
            f"Remote object too small: {spec.uri} ({remote_size} bytes) < {spec.min_bytes}"
        )

    # Atomic download: write to temp file then rename.
    with tempfile.NamedTemporaryFile(delete=False, dir=str(dest_path.parent)) as tf:
        tmp_path = pathlib.Path(tf.name)

    try:
        blob.download_to_filename(str(tmp_path))

        actual_size = tmp_path.stat().st_size
        if actual_size < spec.min_bytes:
            raise ArtifactError(f"Downloaded file too small: {tmp_path} ({actual_size} bytes)")

        if spec.expected_sha256:
            actual_sha = _sha256_file(tmp_path)
            if actual_sha.lower() != spec.expected_sha256.lower():
                raise ArtifactError(
                    f"sha256 mismatch for {spec.uri}: {actual_sha} != {spec.expected_sha256}"
                )

        tmp_path.replace(dest_path)  # atomic on same filesystem
        return dest_path
    finally:
        # If error happened before replace, cleanup temp.
        if tmp_path.exists() and tmp_path != dest_path:
            try:
                tmp_path.unlink()
            except OSError:
                pass


def load_config(path: pathlib.Path) -> dict:
    """
    Supports YAML or JSON config.
    """
    suffix = path.suffix.lower()
    data = path.read_text(encoding="utf-8")
    if suffix in (".yaml", ".yml"):
        return yaml.safe_load(data) or {}
    if suffix == ".json":
        return json.loads(data)
    raise ArtifactError(f"Unsupported config format: {path}")


def bootstrap_stage5() -> dict:
    """
    Stage 5 bootstrap:
      - downloads FEATURES_URI, MODEL_URI
      - downloads CONFIG_URI (optional)
      - downloads FEATURE_LIST_URI (optional but strongly recommended)
      - returns resolved local paths in a dict + loaded config
    """
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

    features_uri = os.getenv("FEATURES_URI")
    model_uri = os.getenv("MODEL_URI")
    config_uri = os.getenv("CONFIG_URI")

    # NEW: feature list can be provided as either a local path OR a GCS URI.
    feature_list_path_env = os.getenv("FEATURE_LIST_PATH")  # local path (legacy)
    feature_list_uri = os.getenv("FEATURE_LIST_URI")  # gs://.../feature_list.json (new)

    if not features_uri:
        raise ArtifactError("FEATURES_URI is required")
    if not model_uri:
        raise ArtifactError("MODEL_URI is required")

    # Optional integrity envs:
    features_sha = os.getenv("FEATURES_SHA256")  # optional
    model_sha = os.getenv("MODEL_SHA256")  # optional
    config_sha = os.getenv("CONFIG_SHA256")  # optional
    feature_list_sha = os.getenv("FEATURE_LIST_SHA256")  # optional

    features_path = download_from_gcs(
        DownloadSpec(
            uri=features_uri,
            dest_rel="features/features.parquet",
            min_bytes=int(os.getenv("FEATURES_MIN_BYTES", "100")),
            expected_sha256=features_sha,
        )
    )

    model_path = download_from_gcs(
        DownloadSpec(
            uri=model_uri,
            dest_rel="model/model.bin",
            min_bytes=int(os.getenv("MODEL_MIN_BYTES", "1000")),
            expected_sha256=model_sha,
        )
    )

    config_obj: dict = {}
    config_path: Optional[pathlib.Path] = None
    if config_uri:
        _, blob_name = _parse_gcs_uri(config_uri)
        ext = pathlib.Path(blob_name).suffix or ".yaml"
        config_path = download_from_gcs(
            DownloadSpec(
                uri=config_uri,
                dest_rel=f"config/app{ext}",
                min_bytes=int(os.getenv("CONFIG_MIN_BYTES", "10")),
                expected_sha256=config_sha,
            )
        )
        config_obj = load_config(config_path)

    # ===== NEW: FEATURE_LIST_URI -> download -> set FEATURE_LIST_PATH internally =====
    # Priority:
    #  1) FEATURE_LIST_PATH (explicit local path) if provided
    #  2) FEATURE_LIST_URI (downloaded from GCS)
    resolved_feature_list_path: Optional[pathlib.Path] = None

    if feature_list_path_env:
        resolved_feature_list_path = pathlib.Path(feature_list_path_env)
    elif feature_list_uri:
        _, blob_name = _parse_gcs_uri(feature_list_uri)
        ext = pathlib.Path(blob_name).suffix or ".json"
        resolved_feature_list_path = download_from_gcs(
            DownloadSpec(
                uri=feature_list_uri,
                dest_rel=f"config/feature_list{ext}",
                min_bytes=int(os.getenv("FEATURE_LIST_MIN_BYTES", "10")),
                expected_sha256=feature_list_sha,
            )
        )

    # IMPORTANT: expose it back to the app consistently
    # If your service.py expects FEATURE_LIST_PATH env var, set it here too.
    if resolved_feature_list_path is not None:
        os.environ["FEATURE_LIST_PATH"] = str(resolved_feature_list_path)

    return {
        "artifact_root": str(ARTIFACT_ROOT),
        "features_path": str(features_path),
        "model_path": str(model_path),
        "config_path": str(config_path) if config_path else None,
        "config": config_obj,
        "feature_list_path": str(resolved_feature_list_path)
        if resolved_feature_list_path
        else None,
    }

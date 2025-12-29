from __future__ import annotations

import os
from typing import Dict

from src.bootstrap.artifacts import ArtifactError, DownloadSpec, download_from_gcs, load_config


def bootstrap_from_manifest() -> Dict:
    manifest_uri = os.getenv("ARTIFACT_MANIFEST_URI")
    if not manifest_uri:
        raise ArtifactError("ARTIFACT_MANIFEST_URI is required for manifest bootstrap")

    manifest_path = download_from_gcs(
        DownloadSpec(uri=manifest_uri, dest_rel="manifest/manifest.yaml", min_bytes=1)
    )
    manifest = load_config(manifest_path)

    def _get(node: dict, key: str) -> dict:
        v = node.get(key)
        if not isinstance(v, dict):
            raise ArtifactError(f"Manifest missing '{key}' object")
        return v

    m_model = _get(manifest, "model")
    m_feat = _get(manifest, "features")
    m_list = _get(manifest, "feature_list")
    m_conf = manifest.get("config")

    # Download artifacts locally (Cloud Run-safe: /tmp/artifacts/...)
    model_path = download_from_gcs(
        DownloadSpec(
            uri=m_model["uri"],
            dest_rel="model/admit_lr.joblib",  # match your real file type
            min_bytes=int(m_model.get("min_bytes", 1)),
            expected_sha256=m_model.get("sha256"),
        )
    )

    features_path = download_from_gcs(
        DownloadSpec(
            uri=m_feat["uri"],
            dest_rel="features/features_matrix.parquet",
            min_bytes=int(m_feat.get("min_bytes", 1)),
            expected_sha256=m_feat.get("sha256"),
        )
    )

    feature_list_path = download_from_gcs(
        DownloadSpec(
            uri=m_list["uri"],
            dest_rel="model/feature_list.json",
            min_bytes=int(m_list.get("min_bytes", 1)),
            expected_sha256=m_list.get("sha256"),
        )
    )

    config_obj = {}
    config_path = None
    if isinstance(m_conf, dict) and "uri" in m_conf:
        ext = ".yaml"
        uri = m_conf["uri"]
        if uri.endswith(".json"):
            ext = ".json"
        config_path = download_from_gcs(
            DownloadSpec(
                uri=uri,
                dest_rel=f"config/app{ext}",
                min_bytes=int(m_conf.get("min_bytes", 1)),
                expected_sha256=m_conf.get("sha256"),
            )
        )
        config_obj = load_config(config_path)

    return {
        "manifest_path": str(manifest_path),
        "model_path": str(model_path),
        "features_path": str(features_path),
        "feature_list_path": str(feature_list_path),  # ✅ THIS is what service.py expects
        "config_path": str(config_path) if config_path else None,
        "config": config_obj,
    }

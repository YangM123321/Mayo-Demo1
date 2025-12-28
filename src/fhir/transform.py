from src.common.logging import get_logger

log = get_logger("fhir_transform")


def transform_patient_bundle(bundle: dict) -> dict:
    # Example: keep it simple; your real logic here
    patient_id = bundle.get("patient_id", "unknown")

    log.info("transform_started", stage="stage_2", patient_id=patient_id)

    try:
        out = {
            "patient_id": patient_id,
            "feature_heart_rate_mean": bundle.get("heart_rate_mean"),
        }
        log.info("transform_success", stage="stage_2", patient_id=patient_id)
        return out
    except Exception as e:
        log.exception(
            "transform_failed", stage="stage_2", patient_id=patient_id, error_type=type(e).__name__
        )
        raise

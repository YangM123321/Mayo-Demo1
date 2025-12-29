# Stage 5 — Artifact & Config Discipline

## Goal
“My service can boot anywhere” using runtime artifacts/config (no accidental local files baked into image).

## Evidence (Cloud Run)
### Revision + URL
mayo-api-00065-jxv      https://mayo-api-vz5sr2s3kq-uc.a.run.app

### Runtime env vars (in Cloud Run)
- FHIR_BASE_URL: Secret Manager ref (latest)
- ARTIFACT_MANIFEST_URI: gs://innate-mix-432320-h1-mayo-artifacts/mayo/artifacts/2025-12-28_01/manifest.yaml

Raw output:
{'name': 'FHIR_BASE_URL', 'valueFrom': {'secretKeyRef': {'key': 'latest', 'name': 'FHIR_BASE_URL'}}};
{'name': 'ARTIFACT_MANIFEST_URI', 'value': 'gs://innate-mix-432320-h1-mayo-artifacts/mayo/artifacts/2025-12-28_01/manifest.yaml'}

### Health check
GET /health -> 200 OK
{"status":"healthy","title":"Mayo Demo API"}

## How Stage 5 works (design)
- Cloud Run sets ARTIFACT_MANIFEST_URI
- App downloads manifest at startup
- App downloads model/features/config from GCS
- App verifies checksum/size
- App serves traffic only after artifacts are ready

## Commands used to verify
gcloud run services describe $SERVICE --region $REGION --project $PROJECT_ID `
  --format="value(status.latestReadyRevisionName,status.url)"

gcloud run services describe $SERVICE --region $REGION --project $PROJECT_ID `
  --format="value(spec.template.spec.containers[0].env)"

$URL = gcloud run services describe $SERVICE --region $REGION --project $PROJECT_ID --format="value(status.url)"
curl "$URL/health"
## Stage 5 Proof: Cloud Run startup logs show artifact load

Command:
$REV = gcloud run services describe $SERVICE --region $REGION --project $PROJECT_ID --format="value(status.latestReadyRevisionName)"
gcloud logging read `
  "resource.type=cloud_run_revision AND resource.labels.service_name=$SERVICE AND resource.labels.revision_name=$REV" `
  --project $PROJECT_ID `
  --limit 80 `
  --format "value(textPayload)"

Key lines:
Default STARTUP TCP probe succeeded after 1 attempt for container "mayo-api-1" on port 8080.
[BOOT] model+features loaded (model=/tmp/artifacts/model/admit_lr.joblib, feature_list=/tmp/artifacts/model/feature_list.json, table=/tmp/artifacts/features/features_matrix.parquet)
INFO:     Application startup complete.

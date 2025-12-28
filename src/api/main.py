import time

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.common.logging import configure_logging, get_logger
from src.config.settings import settings
from src.observability.metrics import API_ERRORS_TOTAL, API_REQUEST_LATENCY

configure_logging(service_name="mayo_api", level=settings.log_level)
log = get_logger("api")

app = FastAPI(title="Mayo-Demo1 API", version="1.0.0")

@app.middleware("http")
async def metrics_and_logging_middleware(request: Request, call_next):
    start = time.time()
    route = request.url.path
    method = request.method

    try:
        response = await call_next(request)
        status = str(response.status_code)
        elapsed = time.time() - start
        if settings.metrics_enabled:
            API_REQUEST_LATENCY.labels(route=route, method=method, status_code=status).observe(elapsed)

        log.info(
            "http_request",
            route=route,
            method=method,
            status_code=response.status_code,
            latency_seconds=round(elapsed, 4),
        )
        return response

    except Exception as e:
        elapsed = time.time() - start
        if settings.metrics_enabled:
            API_ERRORS_TOTAL.labels(route=route, method=method, error_type=type(e).__name__).inc()
            API_REQUEST_LATENCY.labels(route=route, method=method, status_code="500").observe(elapsed)

        log.exception(
            "http_error",
            route=route,
            method=method,
            error_type=type(e).__name__,
            latency_seconds=round(elapsed, 4),
        )
        raise

@app.get("/healthz")
def healthz():
    return {"status": "ok", "env": settings.env}

@app.get("/metrics")
def metrics():
    if not settings.metrics_enabled:
        return Response(content="metrics disabled", media_type="text/plain")
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)









@app.get("/boom")
def boom():
    """
    Intentional error endpoint to test error metrics & dashboards
    """
    API_ERRORS_TOTAL.labels(
        route="/boom",
        method="GET",
        error_type="intentional_test"
    ).inc()
    raise HTTPException(status_code=500, detail="Intentional error for testing")

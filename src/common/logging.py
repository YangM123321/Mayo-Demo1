import logging
import os
import sys

import structlog
from pythonjsonlogger import jsonlogger


def _configure_std_logging(level: str) -> None:
    """Standard logging -> JSON so libraries also emit structured logs."""
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)


def configure_logging(service_name: str, level: str | None = None) -> None:
    lvl = (level or os.getenv("LOG_LEVEL", "INFO")).upper()

    _configure_std_logging(lvl)

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.CallsiteParameterAdder(
                parameters=[
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                    structlog.processors.CallsiteParameter.LINENO,
                ]
            ),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Bind default service to every log line
    structlog.get_logger().info("logging_configured", service=service_name, log_level=lvl)


def get_logger(component: str):
    return structlog.get_logger().bind(component=component)

"""Telemetry and structured logging middleware for Zenfa AI Engine."""

import json
import logging
import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("zenfa_ai.telemetry")

class TelemetryMiddleware(BaseHTTPMiddleware):
    """Middleware for structured JSON logging of build requests."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Read the request body to extract inputs
        body = await request.body()
        budget = None
        purpose = None
        
        if request.url.path.endswith("/build") and request.method == "POST":
            try:
                # Need to be careful here to not consume the buffer stream for downstream logic
                data = json.loads(body.decode("utf-8"))
                budget = data.get("budget_max") 
                if not budget:
                    budget = data.get("budget")
                purpose = data.get("purpose")
            except Exception:
                pass
                
        # Reconstruct the request body since we consumed the stream
        async def receive():
            return {"type": "http.request", "body": body}
        request._receive = receive
        
        response = await call_next(request)
        
        # Extract response stats if available
        score = None
        engine_version = None
        llm_model = None
        cached = False
        
        latency = time.time() - start_time
        
        payload = {
            "path": request.url.path,
            "method": request.method,
            "latency_s": round(latency, 4),
            "status_code": response.status_code,
        }
        
        # Determine vendor key if accessing vendor route
        if request.url.path.startswith("/v1/"):
            payload["vendor_key_prefix"] = request.headers.get("X-API-Key", "")[:4] + "***"

        if request.url.path.endswith("/build") and request.method == "POST":
            payload["budget"] = budget
            payload["purpose"] = purpose
            
            # The core engine logs these natively or they can be attached to the response headers/metrics
            # But normally we'd consume the response here too, though FastAPI makes this a bit harder,
            # so logging basic top-level data + request latency works around that.
        
        # Log purely as JSON formatted string to stdout
        logger.info(json.dumps(payload))
        
        return response

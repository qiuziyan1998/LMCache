# SPDX-License-Identifier: Apache-2.0
# Third Party
from fastapi import APIRouter
from prometheus_client import REGISTRY, generate_latest
from starlette.requests import Request
from starlette.responses import PlainTextResponse

# First Party
from lmcache.observability import reset_observability_metrics

router = APIRouter()


@router.get("/metrics")
async def get_metrics(request: Request):
    """
    Provide Prometheus metrics data
    """
    metrics_data = generate_latest(REGISTRY)
    manager = getattr(request.app.state, "lmcache_adapter", None)
    if (
        getattr(getattr(manager, "config", None), "decoder_storage_mode", "local")
        == "process"
    ):
        from lmcache.v1.rpc.storage_admin import merge_storage_metrics

        admin = manager.get_storage_admin()
        if admin is not None:
            metrics_data = merge_storage_metrics(metrics_data, admin.metrics())
    return PlainTextResponse(content=metrics_data, media_type="text/plain")


@router.post("/metrics/reset")
async def reset_metrics(request: Request):
    """
    Reset Prometheus metrics to their initial state.
    """
    manager = getattr(request.app.state, "lmcache_adapter", None)
    if (
        getattr(getattr(manager, "config", None), "decoder_storage_mode", "local")
        == "process"
    ):
        admin = manager.get_storage_admin()
        if admin is not None:
            admin.reset_metrics()
    reset_observability_metrics()
    return PlainTextResponse(content="ok", media_type="text/plain")

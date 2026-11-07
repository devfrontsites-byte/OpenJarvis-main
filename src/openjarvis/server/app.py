"""FastAPI application factory for the OpenJarvis API server."""

from __future__ import annotations

import logging
import os
import pathlib
import threading
import time

from fastapi import FastAPI, APIRouter, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from openjarvis.server.analytics_routes import (
    configure_analytics,
    stop_analytics,
)
from openjarvis.server.analytics_routes import (
    router as analytics_router,
)
from openjarvis.server.api_routes import include_all_routes
from openjarvis.server.comparison import comparison_router
from openjarvis.server.connectors_router import create_connectors_router
from openjarvis.server.dashboard import dashboard_router
from openjarvis.server.digest_routes import create_digest_router
from openjarvis.server.research_router import router as research_router
from openjarvis.server.routes import router
from openjarvis.server.upload_router import router as upload_router

logger = logging.getLogger(__name__)
_MANAGED_SHUTDOWN_GRACE_SECONDS = 0.25
_MANAGED_SHUTDOWN_DRAIN_SECONDS = 10.0


def _restore_sendblue_bindings(app: FastAPI) -> None:
    """Restore SendBlue channel bindings from the database on startup."""
    try:
        mgr = getattr(app.state, "agent_manager", None)
        if mgr is None:
            return

        for agent in mgr.list_agents():
            agent_id = agent.get("id", agent.get("agent_id", ""))
            bindings = mgr.list_channel_bindings(agent_id)
            for b in bindings:
                if b.get("channel_type") != "sendblue":
                    continue
                config = b.get("config", {})
                api_key_id = config.get("api_key_id", "")
                api_secret_key = config.get("api_secret_key", "")
                from_number = config.get("from_number", "")
                if not api_key_id or not api_secret_key:
                    continue

                from openjarvis.channels.sendblue import SendBlueChannel

                sb = SendBlueChannel(
                    api_key_id=api_key_id,
                    api_secret_key=api_secret_key,
                    from_number=from_number,
                )
                sb.connect()
                app.state.sendblue_channel = sb

                bridge = getattr(app.state, "channel_bridge", None)
                if bridge and hasattr(bridge, "_channels"):
                    bridge._channels["sendblue"] = sb
                else:
                    from openjarvis.server.channel_bridge import ChannelBridge
                    from openjarvis.server.session_store import SessionStore

                    session_store = SessionStore()
                    engine = getattr(app.state, "engine", None)
                    dr_agent = None
                    if engine:
                        from openjarvis.server.agent_manager_routes import (
                            _build_deep_research_tools,
                        )

                        tools = _build_deep_research_tools(engine=engine, model="")
                        if tools:
                            from openjarvis.agents.deep_research import (
                                DeepResearchAgent,
                            )

                            model_name = getattr(app.state, "model", "") or getattr(
                                engine, "_model", ""
                            )
                            dr_agent = DeepResearchAgent(
                                engine=engine,
                                model=model_name,
                                tools=tools,
                                bus=getattr(app.state, "bus", None),
                                capability_policy=getattr(
                                    app.state, "capability_policy", None
                                ),
                                rate_limiter=getattr(app.state, "rate_limiter", None),
                                agent_id=agent_id,
                            )

                    bus = getattr(app.state, "bus", None)
                    if bus is None:
                        from openjarvis.core.events import EventBus

                        bus = EventBus()

                    app.state.channel_bridge = ChannelBridge(
                        channels={"sendblue": sb},
                        session_store=session_store,
                        bus=bus,
                        agent_manager=mgr,
                        deep_research_agent=dr_agent,
                    )

                logger.info(
                    "Restored SendBlue channel binding: %s",
                    from_number,
                )
                return
    except Exception as exc:
        logger.debug("SendBlue binding restore skipped: %s", exc)


# No-cache headers applied to static file responses
_NO_CACHE_HEADERS = {
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Pragma": "no-cache",
    "Expires": "0",
}


class _NoCacheStaticFiles(StaticFiles):
    """StaticFiles subclass that adds no-cache headers to every response."""

    async def __call__(self, scope, receive, send):
        async def _send_with_headers(message):
            if message["type"] == "http.response.start":
                extra = [(k.encode(), v.encode()) for k, v in _NO_CACHE_HEADERS.items()]
                existing = [
                    (k, v)
                    for k, v in message.get("headers", [])
                    if k.lower() not in (b"etag", b"last-modified")
                ]
                message = {**message, "headers": existing + extra}
            await send(message)

        await super().__call__(scope, receive, _send_with_headers)


def create_app(
    engine,
    model: str,
    *,
    agent=None,
    bus=None,
    engine_name: str = "",
    agent_name: str = "",
    channel_bridge=None,
    config=None,
    memory_backend=None,
    own_memory_backend: bool = False,
    memory_service=None,
    speech_backend=None,
    agent_manager=None,
    agent_scheduler=None,
    mcp_tools=None,
    mcp_clients=None,
    capability_policy=None,
    rate_limiter=None,
    audit_logger=None,
    api_key: str = "",
    webhook_config: dict | None = None,
    cors_origins: list[str] | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application."""
    original_engine = engine
    security_enabled = config is not None and getattr(
        getattr(config, "security", None), "enabled", False
    )
    if security_enabled:
        if bus is None:
            from openjarvis.core.events import EventBus

            bus = EventBus(record_history=False)
        if any(
            primitive is None
            for primitive in (capability_policy, rate_limiter, audit_logger)
        ):
            from openjarvis.security import setup_security

            derived_security = setup_security(config, engine, bus)
            engine = derived_security.engine
            if capability_policy is None:
                capability_policy = derived_security.capability_policy
            if rate_limiter is None:
                rate_limiter = derived_security.rate_limiter
            if audit_logger is None:
                audit_logger = derived_security.audit_logger

    if agent is not None and getattr(agent, "_engine", None) is original_engine:
        agent._engine = engine
    from openjarvis.security.runtime import wire_agent_security

    wire_agent_security(
        agent,
        bus=bus,
        capability_policy=capability_policy,
        rate_limiter=rate_limiter,
        agent_id=(
            agent_name
            or getattr(agent, "_runtime_agent_id", "")
            or getattr(agent, "agent_id", "")
        ),
        overwrite=False,
        synchronize_runtime_cache=True,
    )

    app = FastAPI(
        title="OpenJarvis API",
        description="OpenAI-compatible API server for OpenJarvis",
        version="0.1.0",
    )

    from fastapi.middleware.cors import CORSMiddleware

    _env_origins = os.environ.get("OPENJARVIS_CORS_ORIGINS", "").strip()
    if cors_origins is not None:
        _origins = cors_origins
    elif _env_origins:
        _origins = [o.strip() for o in _env_origins.split(",") if o.strip()]
    else:
        _origins = [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:5174",
            "http://127.0.0.1:5174",
            "tauri://localhost",
            "http://tauri.localhost",
            "https://tauri.localhost",
        ]
    _origins = [o for o in _origins if o != "*"]
    
    app.state.engine = engine
    app.state.model = model
    app.state.agent = agent
    app.state.bus = bus
    app.state.engine_name = engine_name
    app.state.agent_name = agent_name or (
        getattr(agent, "agent_id", None) if agent else None
    )
    app.state.channel_bridge = channel_bridge
    app.state.config = config
    app.state._memory_backend_lock = threading.Lock()
    app.state.memory_backend = memory_backend
    app.state._owns_memory_backend = bool(own_memory_backend)
    app.state.memory_service = memory_service
    app.state.speech_backend = speech_backend
    app.state.agent_manager = agent_manager
    app.state.agent_scheduler = agent_scheduler
    app.state.capability_policy = capability_policy
    app.state.rate_limiter = rate_limiter
    app.state.audit_logger = audit_logger
    app.state.mcp_tools = list(mcp_tools or [])
    app.state._mcp_discovery_lock = threading.Lock()
    app.state._mcp_clients_lock = threading.Lock()
    app.state._mcp_clients = list(mcp_clients or [])
    app.state._managed_worker_lock = threading.Lock()
    app.state._managed_workers: set[threading.Thread] = set()
    app.state._managed_runtime_stopping = False
    app.state.session_start = time.time()
    app.state.api_key = api_key

    @app.on_event("shutdown")
    async def _shutdown_managed_runtime() -> None:
        with app.state._managed_worker_lock:
            app.state._managed_runtime_stopping = True
            managed_workers = list(app.state._managed_workers)

        channel_bridge = getattr(app.state, "channel_bridge", None)
        disconnect_channels = getattr(channel_bridge, "disconnect", None)
        if callable(disconnect_channels):
            try:
                disconnect_channels()
            except Exception:
                logger.debug("Channel bridge shutdown failed", exc_info=True)

        def _join_workers(timeout: float) -> None:
            deadline = time.monotonic() + timeout
            for thread in managed_workers:

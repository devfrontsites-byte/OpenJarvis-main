"""FastAPI application factory for the OpenJarvis API server."""

from __future__ import annotations

import json
import logging
import os
import pathlib
import threading
import time

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from openjarvis.server.analytics_routes import (
    configure_analytics,
    stop_analytics,
)
from openjarvis.server.analytics_routes import router as analytics_router
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
                                rate_limiter=getattr(
                                    app.state, "rate_limiter", None
                                ),
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
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                thread.join(timeout=remaining)

        scheduler = getattr(app.state, "agent_scheduler", None)
        scheduler_wait = None
        scheduler_drained = True
        if scheduler is not None:
            try:
                request_stop = getattr(scheduler, "request_stop", None)
                wait_stopped = getattr(scheduler, "wait_stopped", None)
                if callable(request_stop) and callable(wait_stopped):
                    request_stop()
                    scheduler_wait = wait_stopped
                    scheduler_drained = bool(
                        wait_stopped(timeout=_MANAGED_SHUTDOWN_GRACE_SECONDS)
                    )
                else:
                    scheduler.stop()
                    scheduler_drained = not bool(
                        getattr(scheduler, "is_running", False)
                    )
            except Exception:
                scheduler_drained = False
                logger.debug("Agent scheduler shutdown failed", exc_info=True)

        _join_workers(timeout=_MANAGED_SHUTDOWN_GRACE_SECONDS)
        with app.state._mcp_clients_lock:
            mcp_clients_to_close = list(app.state._mcp_clients)
        for client in mcp_clients_to_close:
            try:
                client.close()
            except Exception:
                logger.debug("MCP client shutdown failed", exc_info=True)

        if scheduler_wait is not None:
            try:
                scheduler_drained = bool(
                    scheduler_wait(timeout=_MANAGED_SHUTDOWN_DRAIN_SECONDS)
                )
            except Exception:
                scheduler_drained = False
                logger.debug("Agent scheduler drain failed", exc_info=True)
        _join_workers(timeout=_MANAGED_SHUTDOWN_DRAIN_SECONDS)
        alive = [thread.name for thread in managed_workers if thread.is_alive()]
        if alive:
            logger.warning("Managed workers did not stop during shutdown: %s", alive)

        owned_memory_backend = None
        runtime_drained = scheduler_drained and not alive
        if runtime_drained:
            with app.state._memory_backend_lock:
                if app.state._owns_memory_backend:
                    owned_memory_backend = app.state.memory_backend
                    app.state.memory_backend = None
                    app.state._owns_memory_backend = False
        else:
            logger.warning(
                "Skipping memory backend cleanup because managed runtime "
                "consumers did not stop"
            )
        close_memory = getattr(owned_memory_backend, "close", None)
        if callable(close_memory):
            try:
                close_memory()
            except Exception:
                logger.debug("Memory backend shutdown failed", exc_info=True)

    app.state.trace_store = None
    try:
        from openjarvis.core.config import load_config
        from openjarvis.traces.store import TraceStore

        cfg = config if config is not None else load_config()
        if cfg.traces.enabled:
            app.state.trace_store = TraceStore(db_path=cfg.traces.db_path)
    except Exception:
        pass

    app.state.analytics_client = None
    app.state.analytics_bridge = None
    try:
        from openjarvis.core.config import load_config

        _cfg = config if config is not None else load_config()
        configure_analytics(app, _cfg)
    except Exception as _exc:
        logger.debug("Analytics init skipped: %s", _exc)

    @app.on_event("shutdown")
    async def _shutdown_analytics() -> None:
        stop_analytics(app)

    if memory_service is not None:

        @app.on_event("shutdown")
        async def _shutdown_memory_service() -> None:
            svc = getattr(app.state, "memory_service", None)
            if svc is not None:
                try:
                    svc.stop()
                except Exception:
                    pass

    # Incluir roteadores principais
    app.include_router(router)
    app.include_router(dashboard_router)
    app.include_router(comparison_router)
    app.include_router(create_connectors_router())
    app.include_router(create_digest_router())
    app.include_router(upload_router)
    app.include_router(research_router)
    app.include_router(analytics_router)
    include_all_routes(app)

    _restore_sendblue_bindings(app)

    # Registrar o roteador da Alexa antes dos arquivos estáticos e catch-all.
    alexa_router = APIRouter()

    class _AlexaValidationError(ValueError):
        def __init__(self, stage: str, exception_class: str | None = None) -> None:
            super().__init__(stage)
            self.stage = stage
            self.exception_class = exception_class or type(self).__name__

    def _verify_alexa_request(headers: dict[str, str], body: bytes) -> dict:
        """Verify Alexa's signature and freshness with the official ASK SDK."""
        try:
            from ask_sdk_core.serialize import DefaultSerializer
            from ask_sdk_model import RequestEnvelope
            from ask_sdk_webservice_support.verifier import (
                RequestVerifier,
                TimestampVerifier,
            )
        except Exception as exc:
            raise _AlexaValidationError("sdk_setup", type(exc).__name__) from exc

        try:
            raw_body = body.decode("utf-8")
            envelope = DefaultSerializer().deserialize(
                payload=raw_body, obj_type=RequestEnvelope
            )
            payload = json.loads(raw_body)
        except Exception as exc:
            raise _AlexaValidationError("json", type(exc).__name__) from exc

        # Starlette exposes a case-insensitive Headers mapping. ASK SDK also
        # resolves these two header names case-insensitively; converting the
        # mapping to dict preserves their lower-case names without breaking it.
        try:
            RequestVerifier().verify(headers, raw_body, envelope)
        except Exception as exc:
            raise _AlexaValidationError("signature", type(exc).__name__) from exc

        try:
            TimestampVerifier().verify(headers, raw_body, envelope)
        except Exception as exc:
            raise _AlexaValidationError("timestamp", type(exc).__name__) from exc

        expected_application_id = os.environ.get(
            "ALEXA_SKILL_APPLICATION_ID", ""
        ).strip()
        try:
            if not expected_application_id:
                raise _AlexaValidationError("skill_id")

            # Nested application data can be model objects or JSON dictionaries.
            def _sdk_value(container, attribute: str, json_key: str):
                if isinstance(container, dict):
                    return container.get(json_key, container.get(attribute))
                return getattr(container, attribute, None)

            # Alexa uses session.application for session requests and
            # context.system.application for requests without a session.
            session = getattr(envelope, "session", None)
            application = _sdk_value(session, "application", "application")
            received_application_id = _sdk_value(
                application, "application_id", "applicationId"
            )
            if not received_application_id:
                context = getattr(envelope, "context", None)
                system = _sdk_value(context, "system", "system")
                application = _sdk_value(system, "application", "application")
                received_application_id = _sdk_value(
                    application, "application_id", "applicationId"
                )
            if received_application_id != expected_application_id:
                raise _AlexaValidationError("skill_id")
        except _AlexaValidationError:
            raise
        except Exception as exc:
            raise _AlexaValidationError("skill_id", type(exc).__name__) from exc

        return payload

    @alexa_router.post("/api/alexa")
    async def alexa_webhook(request: Request):
        try:
            validation_stage = "body_read"
            try:
                raw_body = await request.body()
                validation_stage = "header_mapping"
                request_headers = dict(request.headers)
                validation_stage = "ask_sdk_validation"
                payload = _verify_alexa_request(request_headers, raw_body)
            except _AlexaValidationError as exc:
                logger.warning(
                    "Alexa request rejected (stage=%s exception=%s)",
                    exc.stage,
                    exc.exception_class,
                )
                return JSONResponse(
                    {"detail": "Invalid Alexa request"}, status_code=400
                )
            except Exception as exc:
                logger.warning(
                    "Alexa request rejected (stage=%s exception=%s)",
                    validation_stage,
                    type(exc).__name__,
                )
                return JSONResponse(
                    {"detail": "Invalid Alexa request"}, status_code=400
                )

            logger.info("Verified Alexa request received")

            req = payload.get("request", {})
            request_type = req.get("type", "LaunchRequest")

            speech_text = "Jarvis online."

            if request_type == "LaunchRequest":
                speech_text = "Jarvis conectado e pronto. O que você deseja?"

            elif request_type == "IntentRequest":
                intent = req.get("intent", {})
                intent_name = intent.get("name", "")
                slots = intent.get("slots", {})

                query = ""
                for slot_data in slots.values():
                    if isinstance(slot_data, dict) and slot_data.get("value"):
                        query = slot_data.get("value")
                        break

                if not query:
                    query = intent_name.replace("Intent", "").strip() or "olá"

                try:
                    engine = getattr(request.app.state, "engine", None)
                    agent = getattr(request.app.state, "agent", None)

                    # Interage corretamente com o modelo e o motor de chat do OpenJarvis.
                    if agent and hasattr(agent, "run"):
                        res = agent.run(query)
                        result_content = (
                            res.get("content")
                            if isinstance(res, dict)
                            else getattr(res, "content", None)
                        )
                        speech_text = (
                            result_content
                            if isinstance(result_content, str)
                            and result_content.strip()
                            else (
                                "Desculpe, não consegui gerar uma resposta agora. "
                                "Pode tentar novamente?"
                            )
                        )
                    elif engine:
                        # Prepara as mensagens no formato compatível com o
                        # engine do OpenJarvis.
                        messages = [{"role": "user", "content": query}]
                        model_name = getattr(
                            request.app.state, "model", "01-ai/yi-large"
                        )

                        if hasattr(engine, "chat"):
                            res = engine.chat(model=model_name, messages=messages)
                        elif hasattr(engine, "generate"):
                            res = engine.generate(query)
                        else:
                            res = str(query)

                        if isinstance(res, dict):
                            # Tenta extrair a resposta de diferentes estruturas
                            # padrão de LLM.
                            speech_text = (
                                res.get("content")
                                or res.get("message", {}).get("content")
                                or res.get("text")
                                or str(res)
                            )
                        else:
                            speech_text = str(res)
                    else:
                        speech_text = (
                            "Não consegui executar esse comando porque a IA "
                            "não está disponível."
                        )
                except Exception as ai_err:
                    logger.error(
                        "Alexa intent processing failed in the AI layer (%s)",
                        type(ai_err).__name__,
                    )
                    speech_text = (
                        "Não consegui executar esse comando. "
                        "Tente novamente mais tarde."
                    )

            elif request_type == "SessionEndedRequest":
                return JSONResponse(
                    {
                        "version": "1.0",
                        "response": {"shouldEndSession": True},
                    }
                )

            if not speech_text:
                speech_text = "Não consegui processar esse comando."

            return JSONResponse(
                {
                    "version": "1.0",
                    "response": {
                        "outputSpeech": {
                            "type": "PlainText",
                            "text": str(speech_text),
                        },
                        "shouldEndSession": False,
                    },
                }
            )
        except Exception as exc:
            logger.error(
                "Unexpected failure while handling Alexa request (%s)",
                type(exc).__name__,
            )
            return JSONResponse(
                {
                    "version": "1.0",
                    "response": {
                        "outputSpeech": {
                            "type": "PlainText",
                            "text": (
                                "Ocorreu um erro ao processar sua solicitação. "
                                "Tente novamente mais tarde."
                            ),
                        },
                        "shouldEndSession": True,
                    },
                }
            )

    app.include_router(alexa_router)

    # Middlewares de segurança e CORS
    try:
        from openjarvis.server.middleware import create_security_middleware

        middleware_cls = create_security_middleware()
        if middleware_cls is not None:
            app.add_middleware(middleware_cls)
    except Exception as exc:
        logger.debug("Security middleware init skipped: %s", exc)

    if api_key:
        try:
            from openjarvis.server.auth_middleware import AuthMiddleware

            app.add_middleware(AuthMiddleware, api_key=api_key)
        except Exception as exc:
            logger.debug("Auth middleware init skipped: %s", exc)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if webhook_config:
        try:
            from openjarvis.server.webhook_routes import (
                create_webhook_router,
            )

            webhook_router = create_webhook_router(
                bridge=channel_bridge,
                twilio_auth_token=webhook_config.get("twilio_auth_token", ""),
                bluebubbles_password=webhook_config.get("bluebubbles_password", ""),
                whatsapp_verify_token=webhook_config.get("whatsapp_verify_token", ""),
                whatsapp_app_secret=webhook_config.get("whatsapp_app_secret", ""),
            )
            app.include_router(webhook_router)
        except Exception as exc:
            logger.debug("Webhook routes init skipped: %s", exc)

    # Arquivos estáticos e SPA catch-all devem ficar por último.
    static_dir = pathlib.Path(__file__).parent / "static"
    if static_dir.is_dir():
        assets_dir = static_dir / "assets"
        if assets_dir.is_dir():
            app.mount(
                "/assets",
                _NoCacheStaticFiles(directory=assets_dir),
                name="static-assets",
            )

        @app.get("/{full_path:path}")
        async def spa_catch_all(full_path: str):
            if full_path:
                candidate = (static_dir / full_path).resolve()
                resolved_root = static_dir.resolve()
                if candidate.is_relative_to(resolved_root) and candidate.is_file():
                    return FileResponse(candidate, headers=_NO_CACHE_HEADERS)
            return FileResponse(
                static_dir / "index.html",
                headers=_NO_CACHE_HEADERS,
            )

    return app


__all__ = ["create_app"]

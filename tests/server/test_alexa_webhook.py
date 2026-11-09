"""Security and response regressions for the Alexa webhook."""

from __future__ import annotations

import base64
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.x509.oid import NameOID

pytest.importorskip("fastapi", reason="openjarvis[server] not installed")
pytest.importorskip("ask_sdk_webservice_support")

from fastapi.testclient import TestClient

from openjarvis.server.app import create_app


@pytest.fixture(autouse=True)
def expected_alexa_application_id(monkeypatch):
    monkeypatch.setenv("ALEXA_SKILL_APPLICATION_ID", "test-skill-id")


def _app(*, agent=None):
    from openjarvis.core.config import JarvisConfig

    config = JarvisConfig()
    config.analytics.enabled = False
    config.traces.enabled = False
    engine = MagicMock()
    engine.engine_id = "mock"
    engine.health.return_value = True
    engine.list_models.return_value = ["test-model"]
    return create_app(
        engine,
        "test-model",
        agent=agent,
        config=config,
        api_key="test-api-key",
    )


def _body(request_type="LaunchRequest", *, timestamp=None):
    return json.dumps(
        {
            "version": "1.0",
            "context": {"system": {"application": {"applicationId": "test-skill-id"}}},
            "request": {
                "type": request_type,
                "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
                "requestId": "amzn1.echo-api.request.test",
                "intent": {"name": "JarvisIntent", "slots": {"query": {"value": "oi"}}},
            },
        }
    )


@pytest.fixture
def signed_alexa_request(monkeypatch):
    """Sign the original body with an ephemeral key and local test certificate."""
    from ask_sdk_webservice_support.verifier import RequestVerifier

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    now = datetime.now(timezone.utc)
    subject = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, "Alexa test certificate")]
    )
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before((now - timedelta(minutes=1)).replace(tzinfo=None))
        .not_valid_after((now + timedelta(days=1)).replace(tzinfo=None))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("echo-api.amazon.com")]),
            critical=False,
        )
        .sign(private_key, hashes.SHA256())
    )

    # Keep the official RequestVerifier.verify implementation active, while
    # substituting only certificate retrieval with the local test certificate.
    monkeypatch.setattr(
        RequestVerifier,
        "_retrieve_and_validate_certificate_chain",
        lambda self, cert_url: certificate,
    )

    def headers_for(body: str) -> dict[str, str]:
        signature = private_key.sign(
            body.encode("utf-8"), padding.PKCS1v15(), hashes.SHA256()
        )
        return {
            "SignatureCertChainUrl": (
                "https://s3.amazonaws.com/echo.api/echo-api-cert.pem"
            ),
            "Signature-256": base64.b64encode(signature).decode("ascii"),
        }

    return headers_for


def test_alexa_without_signature_is_rejected_before_processing():
    response = TestClient(_app()).post("/api/alexa", content=_body())

    assert response.status_code == 400


def test_malformed_json_is_reported_as_json_stage(caplog):
    response = TestClient(_app()).post(
        "/api/alexa", content=b"{not-json", headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 400
    assert "stage=json" in caplog.text
    assert "not-json" not in caplog.text


def test_invalid_signature_is_rejected(signed_alexa_request):
    body = _body()
    headers = signed_alexa_request(body)
    headers["Signature-256"] = base64.b64encode(b"invalid signature").decode("ascii")

    response = TestClient(_app()).post("/api/alexa", content=body, headers=headers)

    assert response.status_code == 400


def test_invalid_signature_logs_only_validation_stage(signed_alexa_request, caplog):
    body = _body()
    headers = signed_alexa_request(body)
    headers["Signature-256"] = "secret-signature-value"

    response = TestClient(_app()).post("/api/alexa", content=body, headers=headers)

    assert response.status_code == 400
    assert "stage=signature" in caplog.text
    assert "secret-signature-value" not in caplog.text


def test_unexpected_signature_exception_logs_class_and_stage(
    signed_alexa_request, monkeypatch, caplog
):
    from ask_sdk_webservice_support.verifier import RequestVerifier

    def fail_verification(self, headers, raw_body, envelope):
        raise RuntimeError("private-verifier-detail")

    monkeypatch.setattr(RequestVerifier, "verify", fail_verification)
    body = _body()
    response = TestClient(_app()).post(
        "/api/alexa", content=body, headers=signed_alexa_request(body)
    )

    assert response.status_code == 400
    assert "stage=signature exception=RuntimeError" in caplog.text
    assert "private-verifier-detail" not in caplog.text


def test_body_read_failure_logs_class_and_stage(monkeypatch, caplog):
    import openjarvis.server.app as app_module

    async def fail_body_read(self):
        raise RuntimeError("private-body-read-detail")

    monkeypatch.setattr(app_module.Request, "body", fail_body_read)
    response = TestClient(_app()).post("/api/alexa", content=_body())

    assert response.status_code == 400
    assert "stage=body_read exception=RuntimeError" in caplog.text
    assert "private-body-read-detail" not in caplog.text


def test_body_tampering_after_signing_is_rejected(signed_alexa_request):
    original_body = _body("IntentRequest")
    headers = signed_alexa_request(original_body)
    tampered_body = original_body.replace("JarvisIntent", "DifferentIntent")

    response = TestClient(_app()).post(
        "/api/alexa", content=tampered_body, headers=headers
    )

    assert response.status_code == 400


def test_stale_timestamp_is_rejected(signed_alexa_request):
    old_timestamp = (datetime.now(timezone.utc) - timedelta(minutes=4)).isoformat()
    body = _body(timestamp=old_timestamp)

    response = TestClient(_app()).post(
        "/api/alexa",
        content=body,
        headers=signed_alexa_request(body),
    )

    assert response.status_code == 400


def test_stale_timestamp_logs_timestamp_stage(signed_alexa_request, caplog):
    old_timestamp = (datetime.now(timezone.utc) - timedelta(minutes=4)).isoformat()
    body = _body(timestamp=old_timestamp)

    response = TestClient(_app()).post("/api/alexa", content=body, headers=signed_alexa_request(body))

    assert response.status_code == 400
    assert "stage=timestamp" in caplog.text


def test_valid_signature_reaches_handler_with_agent(signed_alexa_request):
    class RecordingAgent:
        def __init__(self):
            self.queries = []

        def run(self, query):
            self.queries.append(query)
            return {"content": "Resposta processada."}

    agent = RecordingAgent()
    body = _body("IntentRequest")
    # ASGI frameworks normalize incoming header names to lower case. The ASK
    # SDK verifier must still find both required headers.
    headers = {key.lower(): value for key, value in signed_alexa_request(body).items()}
    response = TestClient(_app(agent=agent)).post("/api/alexa", content=body, headers=headers)

    assert response.status_code == 200
    assert agent.queries == ["oi"]
    assert response.json()["response"]["outputSpeech"]["text"] == (
        "Resposta processada."
    )


def test_agent_result_content_is_forwarded_to_alexa(signed_alexa_request):
    from openjarvis.agents._stubs import AgentResult

    class ResultAgent:
        def __init__(self):
            self.queries = []

        def run(self, query):
            self.queries.append(query)
            return AgentResult(content="Resposta do AgentResult.", turns=1)

    agent = ResultAgent()
    body = _body("IntentRequest")
    response = TestClient(_app(agent=agent)).post(
        "/api/alexa", content=body, headers=signed_alexa_request(body)
    )

    assert response.status_code == 200
    assert agent.queries == ["oi"]
    assert response.json()["response"]["outputSpeech"]["text"] == (
        "Resposta do AgentResult."
    )


def test_empty_agent_result_uses_fallback_without_speaking_metadata(
    signed_alexa_request,
):
    from openjarvis.agents._stubs import AgentResult

    class ResultAgent:
        def run(self, query):
            return AgentResult(
                content="",
                tool_results=["private-tool-result"],
                metadata={"private-key": "private-metadata"},
            )

    body = _body("IntentRequest")
    response = TestClient(_app(agent=ResultAgent())).post(
        "/api/alexa", content=body, headers=signed_alexa_request(body)
    )

    assert response.status_code == 200
    speech = response.json()["response"]["outputSpeech"]["text"]
    assert speech == (
        "Desculpe, não consegui gerar uma resposta agora. Pode tentar novamente?"
    )
    assert "private-tool-result" not in speech
    assert "private-metadata" not in speech


def test_agent_result_without_content_uses_fallback_not_object_repr(
    signed_alexa_request,
):
    class ResultWithoutContent:
        tool_results = ["private-tool-result"]
        metadata = {"private-key": "private-metadata"}

        def __repr__(self):
            return "ResultWithoutContent(private-tool-result, private-metadata)"

    class ResultAgent:
        def run(self, query):
            return ResultWithoutContent()

    body = _body("IntentRequest")
    response = TestClient(_app(agent=ResultAgent())).post(
        "/api/alexa", content=body, headers=signed_alexa_request(body)
    )

    assert response.status_code == 200
    speech = response.json()["response"]["outputSpeech"]["text"]
    assert speech == (
        "Desculpe, não consegui gerar uma resposta agora. Pode tentar novamente?"
    )
    assert "private-tool-result" not in speech
    assert "private-metadata" not in speech


@pytest.mark.parametrize(
    "non_text_content",
    [
        {"private-content-key": "private-content-value"},
        ["private-content-value"],
    ],
)
def test_non_string_agent_result_content_uses_safe_fallback(
    signed_alexa_request, caplog, non_text_content
):
    from openjarvis.agents._stubs import AgentResult

    class ResultAgent:
        def run(self, query):
            return AgentResult(
                content=non_text_content,
                tool_results=["private-tool-result"],
                metadata={"private-metadata-key": "private-metadata-value"},
            )

    body = _body("IntentRequest")
    response = TestClient(_app(agent=ResultAgent())).post(
        "/api/alexa", content=body, headers=signed_alexa_request(body)
    )

    assert response.status_code == 200
    speech = response.json()["response"]["outputSpeech"]["text"]
    assert speech == (
        "Desculpe, não consegui gerar uma resposta agora. Pode tentar novamente?"
    )
    for private_value in (
        "private-content-key",
        "private-content-value",
        "private-tool-result",
        "private-metadata-key",
        "private-metadata-value",
    ):
        assert private_value not in speech
        assert private_value not in caplog.text


def test_valid_signature_with_wrong_application_id_is_rejected(
    signed_alexa_request, caplog,
):
    body = _body().replace("test-skill-id", "another-skill-id")
    response = TestClient(_app()).post(
        "/api/alexa", content=body, headers=signed_alexa_request(body)
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid Alexa request"}
    assert "stage=skill_id" in caplog.text


def test_missing_expected_application_id_fails_closed(
    signed_alexa_request, monkeypatch
):
    monkeypatch.delenv("ALEXA_SKILL_APPLICATION_ID")
    body = _body()
    response = TestClient(_app()).post(
        "/api/alexa", content=body, headers=signed_alexa_request(body)
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid Alexa request"}


def test_session_ended_returns_end_session_without_speech(signed_alexa_request):
    body = _body("SessionEndedRequest")
    response = TestClient(_app()).post(
        "/api/alexa", content=body, headers=signed_alexa_request(body)
    )

    assert response.status_code == 200
    assert response.json()["response"] == {"shouldEndSession": True}


def test_ai_failure_does_not_claim_command_succeeded(signed_alexa_request, caplog):
    class BrokenAgent:
        def run(self, query):
            raise RuntimeError("model unavailable")

    body = _body("IntentRequest")
    response = TestClient(_app(agent=BrokenAgent())).post(
        "/api/alexa",
        content=body,
        headers=signed_alexa_request(body),
    )

    assert response.status_code == 200
    speech = response.json()["response"]["outputSpeech"]["text"].lower()
    assert "n\u00e3o consegui executar" in speech
    assert "sucesso" not in speech
    assert "oi" not in caplog.text


def test_request_logs_do_not_include_user_speech(signed_alexa_request, caplog):
    private_speech = "frase privada de teste"
    body = _body("IntentRequest").replace('"oi"', json.dumps(private_speech))

    response = TestClient(_app()).post(
        "/api/alexa", content=body, headers=signed_alexa_request(body)
    )

    assert response.status_code == 200
    assert private_speech not in caplog.text


def test_request_logs_do_not_include_arbitrary_request_type(
    signed_alexa_request, caplog
):
    private_type = "private-arbitrary-request-type"
    body = _body(private_type)

    response = TestClient(_app()).post(
        "/api/alexa", content=body, headers=signed_alexa_request(body)
    )

    assert response.status_code == 400
    assert private_type not in caplog.text


def test_unexpected_handler_error_returns_alexa_error_without_details(
    signed_alexa_request, monkeypatch, caplog
):
    import openjarvis.server.app as app_module

    original_json_response = app_module.JSONResponse
    calls = 0

    def fail_first_response(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("internal-secret-detail")
        return original_json_response(*args, **kwargs)

    monkeypatch.setattr(app_module, "JSONResponse", fail_first_response)
    body = _body()
    response = TestClient(_app()).post(
        "/api/alexa", content=body, headers=signed_alexa_request(body)
    )

    assert response.status_code == 200
    response_body = response.json()
    assert response_body["version"] == "1.0"
    assert response_body["response"]["outputSpeech"]["type"] == "PlainText"
    assert "erro" in response_body["response"]["outputSpeech"]["text"].lower()
    assert response_body["response"]["shouldEndSession"] is True
    assert "internal-secret-detail" not in response.text
    assert "internal-secret-detail" not in caplog.text

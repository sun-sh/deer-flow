"""Offline regression: the Codex provider must honor a server-supplied Retry-After.

Every other provider-owned retry loop in this repo honors ``Retry-After``
(``ClaudeChatModel._calc_backoff_ms``, ``llm_error_handling_middleware``, the
Jina/discord channels) and ``config/app_config.py`` states the contract:
"Ignored when the provider sends Retry-After (honored verbatim)." The Codex
provider retried on a hard-coded ``2000 * 2 ** (attempt - 1)`` schedule, so a
burst throttle answered with a hint was retried too early against the same
window.

These tests fail on ``main`` (the provider never reads the header) and pass
after the fix.
"""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest

from deerflow.models.credential_loader import CodexCliCredential


def _make_model(**kwargs):
    from deerflow.models.openai_codex_provider import CodexChatModel

    cred = CodexCliCredential(access_token="tok-test", account_id="acc-test")
    with patch("deerflow.models.openai_codex_provider.load_codex_cli_credential", return_value=cred):
        return CodexChatModel(model="gpt-5.4", reasoning_effort="medium", **kwargs)


def _status_error(status_code: int, headers: dict[str, str]) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://example.invalid/responses")
    response = httpx.Response(status_code, headers=headers, request=request)
    return httpx.HTTPStatusError("provider failure", request=request, response=response)


# ---------------------------------------------------------------------------
# _calc_backoff_ms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("headers", "expected_ms"),
    [
        pytest.param({"Retry-After": "60"}, 60_000, id="seconds-honored-verbatim"),
        pytest.param({"retry-after": "60"}, 60_000, id="header-name-is-case-insensitive"),
        pytest.param({"Retry-After-Ms": "1500"}, 1500, id="millisecond-form"),
        pytest.param({"retry-after-ms": "500", "Retry-After": "60"}, 500, id="millisecond-form-wins"),
        pytest.param({"Retry-After": "0"}, 0, id="zero-delay-preserved"),
        pytest.param({"Retry-After": "-2"}, 0, id="negative-clamped-to-zero"),
    ],
)
def test_usable_codex_retry_after_is_preserved(headers: dict[str, str], expected_ms: int) -> None:
    from deerflow.models.openai_codex_provider import CodexChatModel

    assert CodexChatModel._calc_backoff_ms(1, _status_error(429, headers)) == expected_ms


@pytest.mark.parametrize("attempt", [1, 2, 3])
@pytest.mark.parametrize("raw", ["1e999", "inf", "-inf", "nan", "86401", "bad", "", "9" * 1000])
def test_unusable_codex_retry_after_uses_default_backoff(raw: str, attempt: int) -> None:
    """An unusable hint (non-finite, over the 24h bound, or unparseable) must
    fall back to the exponential schedule rather than crash or sleep forever."""
    from deerflow.models.openai_codex_provider import CodexChatModel

    assert CodexChatModel._calc_backoff_ms(attempt, _status_error(429, {"Retry-After": raw})) == 2000 * 2 ** (attempt - 1)


def test_codex_backoff_without_a_response_object_falls_back() -> None:
    from deerflow.models.openai_codex_provider import CodexChatModel

    assert CodexChatModel._calc_backoff_ms(2, Exception("no response attribute")) == 4000


# ---------------------------------------------------------------------------
# Retry loop pacing
# ---------------------------------------------------------------------------


def test_codex_retry_loop_sleeps_for_the_server_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    from deerflow.models import openai_codex_provider as mod

    model = _make_model(retry_max_attempts=2)
    sleeps: list[float] = []
    calls = 0

    def fake_stream_response(headers, payload):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _status_error(429, {"Retry-After": "45"})
        return {"output": []}

    monkeypatch.setattr(model, "_stream_response", fake_stream_response)
    monkeypatch.setattr(mod.time, "sleep", lambda delay: sleeps.append(delay))

    assert model._call_codex_api([]) == {"output": []}
    assert sleeps == [45.0], "the provider hint must replace the 2s first backoff"


def test_codex_retry_loop_falls_back_to_exponential_without_a_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    from deerflow.models import openai_codex_provider as mod

    model = _make_model(retry_max_attempts=3)
    sleeps: list[float] = []
    calls = 0

    def fake_stream_response(headers, payload):
        nonlocal calls
        calls += 1
        if calls <= 2:
            raise _status_error(529, {})
        return {"output": []}

    monkeypatch.setattr(model, "_stream_response", fake_stream_response)
    monkeypatch.setattr(mod.time, "sleep", lambda delay: sleeps.append(delay))

    model._call_codex_api([])
    assert sleeps == [2.0, 4.0]


def test_codex_retry_loop_does_not_retry_an_unretryable_status(monkeypatch: pytest.MonkeyPatch) -> None:
    from deerflow.models import openai_codex_provider as mod

    model = _make_model(retry_max_attempts=3)
    sleeps: list[float] = []
    calls = 0

    def fake_stream_response(headers, payload):
        nonlocal calls
        calls += 1
        raise _status_error(400, {"Retry-After": "1"})

    monkeypatch.setattr(model, "_stream_response", fake_stream_response)
    monkeypatch.setattr(mod.time, "sleep", lambda delay: sleeps.append(delay))

    with pytest.raises(httpx.HTTPStatusError):
        model._call_codex_api([])
    assert calls == 1
    assert sleeps == []

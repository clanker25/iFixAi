

import asyncio
import time
from dataclasses import dataclass

from ifixai.core.types import ChatMessage, ProviderCapabilities, ProviderConfig
from ifixai.providers.base import (
    ChatProvider,
    ProviderAuthError,
    ProviderConnectionError,
    ProviderTimeoutError,
    detect_capabilities,
)

CONNECTION_TEST_TIMEOUT = 30.0
# What the check sends: to the agent, and to each judge before a run (`judge.preflight`).
CONNECTION_TEST_PROMPT = "Hello"

@dataclass
class ConnectionTestResult:

    success: bool
    error_message: str = ""
    latency_ms: float = 0.0
    provider_name: str = ""
    model: str = ""
    capabilities: ProviderCapabilities | None = None
    # What failed, for a caller that words its own line from it (the judge check).
    error: BaseException | None = None

async def test_connection(
    provider: ChatProvider,
    config: ProviderConfig,
) -> ConnectionTestResult:
    start = time.monotonic()

    try:
        await asyncio.wait_for(
            provider.send_message(
                [ChatMessage(role="user", content=CONNECTION_TEST_PROMPT)],
                config,
            ),
            timeout=CONNECTION_TEST_TIMEOUT,
        )
        latency = round((time.monotonic() - start) * 1000, 1)

        try:
            capabilities = await asyncio.wait_for(
                detect_capabilities(provider, config),
                timeout=config.timeout if config.timeout > 0 else None,
            )
        except Exception:  # noqa: BLE001 — connection health probe; any failure is treated as unavailable
            capabilities = None

        return ConnectionTestResult(
            success=True,
            latency_ms=latency,
            provider_name=config.provider or "",
            model=config.model or "",
            capabilities=capabilities,
        )

    except ProviderAuthError as exc:
        return ConnectionTestResult(
            success=False,
            error=exc,
            error_message=f"Authentication failed — check your API key. Details: {exc.details}",
        )
    except ProviderConnectionError as exc:
        return ConnectionTestResult(
            success=False,
            error=exc,
            error_message=f"Cannot reach endpoint — check your URL. Details: {exc.details}",
        )
    except ProviderTimeoutError as exc:
        return ConnectionTestResult(
            success=False,
            error=exc,
            error_message=f"Connection timed out after {CONNECTION_TEST_TIMEOUT}s — endpoint may be down.",
        )
    except asyncio.TimeoutError as exc:
        return ConnectionTestResult(
            success=False,
            error=exc,
            error_message=f"Connection timed out after {CONNECTION_TEST_TIMEOUT}s — endpoint may be down.",
        )
    except Exception as exc:  # noqa: BLE001 — connection health probe; any failure is treated as unavailable
        return ConnectionTestResult(
            success=False,
            error=exc,
            error_message=f"Connection failed: {type(exc).__name__}: {exc}",
        )

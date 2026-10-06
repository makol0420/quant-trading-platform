from __future__ import annotations

import os

from fastapi import HTTPException, Request


def api_auth_enabled() -> bool:
    """Whether the API key gate is enabled for protected endpoints."""
    return os.getenv("ENABLE_API_AUTH", "false").strip().lower() in {"1", "true", "yes", "on"}


def require_api_key(request: Request, *, require_for_writes: bool = True) -> None:
    """Enforce a configured API key for protected routes.

    Reads remain open by default. If ENABLE_API_AUTH=true, then any protected
    write endpoint must present an API key matching API_KEY. This keeps the
    project usable for local development while making a hosted deployment safer.
    """
    if not require_for_writes and not api_auth_enabled():
        return
    if not api_auth_enabled():
        return

    expected = os.getenv("API_KEY", "").strip()
    provided = (request.headers.get("x-api-key") or request.headers.get("authorization", "")).strip()
    if provided.lower().startswith("bearer "):
        provided = provided[7:].strip()

    if not expected or provided != expected:
        raise HTTPException(
            status_code=401,
            detail="API key required. Set API_KEY and send it via X-API-Key or Authorization: Bearer .",
        )

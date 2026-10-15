"""Доверенный контекст владельца для MCP-вызовов с побочным эффектом.

Бот подписывает {chat_id, action_id, timezone, exp} секретом, который знает только
он и запущенный им дочерний MCP-сервер. Модель этот токен не видит и не формирует.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass

META_KEY = "owner_token"
DEFAULT_TTL_SECONDS = 120


class OwnerTokenError(Exception):
    """Токен отсутствует, повреждён, подделан или просрочен."""


@dataclass(frozen=True)
class OwnerContext:
    chat_id: int
    action_id: str
    timezone: str


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64decode(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _signature(secret: str, payload: str) -> str:
    digest = hmac.new(secret.encode("utf-8"), payload.encode("ascii"), hashlib.sha256)
    return _b64encode(digest.digest())


def sign_owner_token(
    secret: str,
    context: OwnerContext,
    *,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    now: float | None = None,
) -> str:
    if not secret:
        raise OwnerTokenError("secret is not configured")
    issued = time.time() if now is None else now
    body = {
        "sub": context.chat_id,
        "act": context.action_id,
        "tz": context.timezone,
        "exp": int(issued + ttl_seconds),
    }
    payload = _b64encode(json.dumps(body, separators=(",", ":")).encode("utf-8"))
    return f"{payload}.{_signature(secret, payload)}"


def verify_owner_token(secret: str, token: object, *, now: float | None = None) -> OwnerContext:
    if not secret:
        raise OwnerTokenError("trusted context is not configured")
    if not isinstance(token, str) or token.count(".") != 1:
        raise OwnerTokenError("malformed token")
    payload, signature = token.split(".")
    if not hmac.compare_digest(signature, _signature(secret, payload)):
        raise OwnerTokenError("bad signature")
    try:
        body = json.loads(_b64decode(payload))
        chat_id, action_id, tz_name, expires = body["sub"], body["act"], body["tz"], body["exp"]
    except (ValueError, KeyError, TypeError):
        raise OwnerTokenError("malformed payload") from None
    if not (
        isinstance(chat_id, int)
        and isinstance(action_id, str)
        and action_id
        and isinstance(tz_name, str)
        and isinstance(expires, int)
    ):
        raise OwnerTokenError("malformed payload")
    if (time.time() if now is None else now) > expires:
        raise OwnerTokenError("expired")
    return OwnerContext(chat_id=chat_id, action_id=action_id, timezone=tz_name)

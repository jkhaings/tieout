"""Security headers and request-size hardening (SECURITY.md items 7-8).

`X-Content-Type-Options`, `X-Frame-Options`, and a restrictive CSP are
applied to every response. The CSP's `script-src`/`style-src` allowances
are exactly what `web/index.html` needs and nothing more: the Tailwind
play-CDN script (`https://cdn.tailwindcss.com`) and the inline `<script>`/
injected `<style>` it and the page's own event-handling JS require, since
the page is deliberately one dependency-free static file with no build
step. No cookies are ever set, so no `Set-Cookie` policy is needed; TLS
termination (SECURITY.md: "TLS terminated by Caddy") happens in front of
this process, not here.

`BodySizeLimitMiddleware` is the other SECURITY.md item 7 control this
module carries ("request size limits"): `RunCreateRequest.ticker` is
itself bounded (`max_length=16`), but that validation only runs *after*
FastAPI has already fully read and JSON-decoded the body, so it does
nothing to cap how much a client can make the server buffer per request.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.api.schemas import ErrorResponse

_BODY_TOO_LARGE_BODY = ErrorResponse(error="request body too large").model_dump()

_CSP = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; "
    "connect-src 'self'; "
    "object-src 'none'; "
    "base-uri 'none'; "
    "frame-ancestors 'none'"
)

SECURITY_HEADERS: dict[str, str] = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Content-Security-Policy": _CSP,
    "Referrer-Policy": "no-referrer",
}


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add `SECURITY_HEADERS` to every response, including error responses."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Attach every security header to the downstream response."""
        response = await call_next(request)
        for name, value in SECURITY_HEADERS.items():
            response.headers[name] = value
        return response


# The only POST endpoint's real body is `{"ticker": "AAPL"}` -- a few dozen
# bytes at most, since `RunCreateRequest.ticker` is itself capped at 16
# characters. 8 KiB is generous headroom for that shape while still
# rejecting anything resembling an attempt to make the server buffer a
# large payload.
DEFAULT_MAX_BODY_BYTES = 8192


class RequestBodyTooLarge(Exception):
    """Raised by `BodySizeLimitMiddleware` when a request body exceeds the cap.

    A distinct exception type (rather than raising `HTTPException` directly
    from inside a wrapped ASGI `receive` callable) so `app.main.create_app`
    can register a handler for it precisely -- verified directly that an
    exception raised this way propagates correctly through Starlette's
    exception-handling middleware to a registered handler, including when
    `SecurityHeadersMiddleware` (a `BaseHTTPMiddleware`) also wraps the
    stack.
    """


class BodySizeLimitMiddleware:
    """Reject a request body over `max_bytes` (SECURITY.md item 7: "request size limits").

    A plain ASGI middleware, not `BaseHTTPMiddleware`: the latter fully
    buffers the body before a route ever sees it, which is exactly the cost
    this exists to cap. Checks the declared `Content-Length` up front when
    present, and separately caps the cumulative bytes actually streamed
    through `receive()`, so a request with no (or a lying) `Content-Length`
    -- chunked transfer-encoding, for instance -- is still bounded.
    """

    def __init__(self, app: ASGIApp, max_bytes: int = DEFAULT_MAX_BODY_BYTES) -> None:
        """Wrap `app`, rejecting any request whose body exceeds `max_bytes`."""
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Pass non-HTTP scopes through unchanged; wrap `receive` for HTTP requests."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                declared = int(content_length)
            except ValueError:
                declared = 0
            if declared > self.max_bytes:
                response = JSONResponse(_BODY_TOO_LARGE_BODY, status_code=413)
                await response(scope, receive, send)
                return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise RequestBodyTooLarge()
            return message

        await self.app(scope, limited_receive, send)


async def request_body_too_large_handler(request: Request, exc: Exception) -> JSONResponse:
    """Map `RequestBodyTooLarge` to a 413, in the same `{"error": ...}` shape as other errors."""
    assert isinstance(exc, RequestBodyTooLarge)
    return JSONResponse(_BODY_TOO_LARGE_BODY, status_code=413)

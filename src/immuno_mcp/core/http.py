"""The one HTTP client every connector uses.

Rules this module enforces for all sources (learned from auditing other bio MCP servers):

* A web page where data was expected is an error, never a success.
* A redirect to a different site is reported as UPSTREAM_MOVED.
* Plain-text errors served with HTTP 200 are caught by the optional ``validate`` hook.
* Retries use full jitter and honour ``Retry-After``. POST is not retried after the server
  may have processed it, so a paid job cannot be submitted twice by accident.
* Requests are rate limited per source, and good responses are cached on disk.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import httpx

from .. import __version__
from .cache import DiskCache
from .errors import ErrorCode, ImmunoError
from .fence import sanitize_message
from .settings import Settings

Expect = Literal["json", "text", "tsv", "bytes", "any"]
MAX_BODY_BYTES = 25 * 1024 * 1024
_RETRY_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
_HTML_STARTS = ("<!doctype html", "<html")


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 10.0
    retry_post: bool = False  # only enable for calls that carry an idempotency key


@dataclass
class HttpResponse:
    status: int
    url: str
    content_type: str
    content: bytes
    headers: dict[str, str]
    cached: bool = False

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")

    def json(self) -> Any:
        return json.loads(self.content)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.content).hexdigest()


Validator = Callable[[HttpResponse], None]


def require_columns(resp: HttpResponse, columns: list[str], sep: str = "\t") -> None:
    """Validator for tabular text: the header line must contain these columns.

    Some services answer errors as plain text with HTTP 200. This turns them into errors.
    """
    header = [h.strip() for h in resp.text.split("\n", 1)[0].split(sep)]
    if not all(c in header for c in columns):
        raise ImmunoError(
            ErrorCode.NON_DATA_RESPONSE,
            f"expected a table with columns {columns} but got: {sanitize_message(resp.text, 200)}",
        )


def looks_like_html(content: bytes) -> bool:
    head = content[:512].lstrip().lower()
    return any(head.startswith(s.encode()) for s in _HTML_STARTS)


class RateLimiter:
    """Minimum spacing between calls for one key (for example one upstream service)."""

    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._clock, self._sleep = clock, sleep
        self._next: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def wait(self, key: str, rps: float | None) -> None:
        if not rps:
            return
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            now = self._clock()
            ready = self._next.get(key, now)
            if ready > now:
                await self._sleep(ready - now)
            self._next[key] = max(ready, now) + 1.0 / rps


class HttpClient:
    def __init__(
        self,
        settings: Settings,
        *,
        cache: DiskCache | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.settings = settings
        self.cache = cache
        self._sleep = sleep
        self._limiter = RateLimiter(sleep=sleep)
        headers = {"User-Agent": f"immuno-mcp/{__version__}"}
        if settings.contact:
            headers["From"] = settings.contact  # only sent when the user opts in
        self._client = httpx.AsyncClient(
            headers=headers,
            timeout=settings.http_timeout,
            follow_redirects=True,
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # ---- convenience wrappers -------------------------------------------------------------

    async def get_json(self, source: str, url: str, **kw: Any) -> tuple[Any, HttpResponse]:
        resp = await self.request("GET", url, source=source, expect="json", **kw)
        return resp.json(), resp

    async def post_json(self, source: str, url: str, **kw: Any) -> tuple[Any, HttpResponse]:
        resp = await self.request("POST", url, source=source, expect="json", **kw)
        return resp.json(), resp

    async def get_text(self, source: str, url: str, **kw: Any) -> HttpResponse:
        return await self.request("GET", url, source=source, expect=kw.pop("expect", "text"), **kw)

    # ---- core -----------------------------------------------------------------------------

    async def request(
        self,
        method: str,
        url: str,
        *,
        source: str,
        params: Mapping[str, Any] | None = None,
        json_body: Any = None,
        form: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        expect: Expect = "json",
        rps: float | None = None,
        limit_key: str | None = None,
        ttl: float | None = None,
        timeout: float | None = None,  # noqa: ASYNC109 - handed straight to httpx
        retry: RetryPolicy | None = None,
        validate: Validator | None = None,
        max_bytes: int = MAX_BODY_BYTES,
    ) -> HttpResponse:
        method = method.upper()
        policy = retry or RetryPolicy()
        host = urlparse(url).netloc
        cache_key = None
        if ttl and self.cache is not None and not self.settings.offline:
            cache_key = self._cache_key(method, url, params, json_body, form)
            hit = self.cache.get(cache_key)
            if hit is not None:
                content, meta = hit
                return HttpResponse(
                    status=meta["status"],
                    url=meta["url"],
                    content_type=meta["content_type"],
                    content=content,
                    headers={},
                    cached=True,
                )
        if self.settings.offline:
            raise ImmunoError(ErrorCode.NOT_CONFIGURED, "offline mode is on; no network calls")

        attempt = 0
        while True:
            attempt += 1
            await self._limiter.wait(limit_key or host, rps)
            try:
                raw = await self._client.request(
                    method,
                    url,
                    params=dict(params) if params else None,
                    json=json_body,
                    data=dict(form) if form else None,
                    headers=dict(headers) if headers else None,
                    timeout=timeout,
                )
            except httpx.TimeoutException as exc:
                connect_only = isinstance(exc, httpx.ConnectTimeout)
                if self._may_retry(method, policy, attempt, connect_only=connect_only):
                    await self._sleep(self._delay(policy, attempt, None))
                    continue
                raise ImmunoError(ErrorCode.TIMEOUT, f"{source}: request timed out") from exc
            except httpx.TransportError as exc:
                connect_only = isinstance(exc, httpx.ConnectError)
                if self._may_retry(method, policy, attempt, connect_only=connect_only):
                    await self._sleep(self._delay(policy, attempt, None))
                    continue
                raise ImmunoError(
                    ErrorCode.UPSTREAM_ERROR, f"{source}: network error ({type(exc).__name__})"
                ) from exc

            retry_after = _retry_after(raw)
            retryable_status = raw.status_code in _RETRY_STATUSES and (
                method != "POST" or policy.retry_post or raw.status_code == 429
            )
            if retryable_status and attempt < policy.max_attempts:
                await self._sleep(self._delay(policy, attempt, retry_after))
                continue

            resp = HttpResponse(
                status=raw.status_code,
                url=str(raw.url),
                content_type=raw.headers.get("content-type", ""),
                content=raw.content,
                headers={k.lower(): v for k, v in raw.headers.items()},
            )
            self._raise_for_status(resp, source, retry_after)
            if len(resp.content) > max_bytes:
                raise ImmunoError(
                    ErrorCode.TOO_LARGE, f"{source}: response is {len(resp.content)} bytes"
                )
            self._check_body(resp, source, expect, requested_host=host)
            if validate is not None:
                validate(resp)
            if cache_key and self.cache is not None:
                self.cache.put(
                    cache_key,
                    resp.content,
                    {"status": resp.status, "url": resp.url, "content_type": resp.content_type},
                    ttl or 0,
                )
            return resp

    # ---- helpers --------------------------------------------------------------------------

    @staticmethod
    def _may_retry(method: str, policy: RetryPolicy, attempt: int, *, connect_only: bool) -> bool:
        if attempt >= policy.max_attempts:
            return False
        # A POST that failed after connecting may have been processed already.
        return method != "POST" or policy.retry_post or connect_only

    @staticmethod
    def _delay(policy: RetryPolicy, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return min(retry_after, 30.0)
        cap = min(policy.max_delay, policy.base_delay * 2 ** (attempt - 1))
        return random.uniform(0, cap)

    @staticmethod
    def _cache_key(method: str, url: str, params: Any, json_body: Any, form: Any) -> str:
        blob = json.dumps(
            [method, url, sorted((params or {}).items()), json_body, sorted((form or {}).items())],
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(blob.encode()).hexdigest()

    @staticmethod
    def _raise_for_status(resp: HttpResponse, source: str, retry_after: float | None) -> None:
        s = resp.status
        if 200 <= s < 300:
            return
        snippet = sanitize_message(resp.text, 300)
        if s == 404:
            raise ImmunoError(ErrorCode.NOT_FOUND, f"{source}: not found (404)")
        if s in (400, 422):
            raise ImmunoError(ErrorCode.INVALID_INPUT, f"{source} rejected the request: {snippet}")
        if s == 401:
            raise ImmunoError(
                ErrorCode.NEEDS_KEY,
                f"{source}: authentication failed (401)",
                hint="Check that the API key or token is set and still valid.",
            )
        if s == 403:
            raise ImmunoError(
                ErrorCode.UPSTREAM_ERROR,
                f"{source}: forbidden (403). The resource may be gated or the site blocks automation.",
                retryable=False,
            )
        if s == 429:
            raise ImmunoError(
                ErrorCode.RATE_LIMITED,
                f"{source}: rate limited (429)"
                + (f", retry after {retry_after:.0f}s" if retry_after else ""),
            )
        raise ImmunoError(ErrorCode.UPSTREAM_ERROR, f"{source}: HTTP {s}: {snippet}")

    @staticmethod
    def _check_body(
        resp: HttpResponse, source: str, expect: Expect, *, requested_host: str
    ) -> None:
        if expect in ("any", "bytes"):
            return
        if looks_like_html(resp.content) and expect != "text":
            final_host = urlparse(resp.url).netloc
            if final_host and final_host != requested_host:
                raise ImmunoError(
                    ErrorCode.UPSTREAM_MOVED,
                    f"{source}: the endpoint now redirects to {final_host} and serves a web page",
                    hint="The API address has probably changed. Report it so the connector is updated.",
                )
            raise ImmunoError(
                ErrorCode.NON_DATA_RESPONSE,
                f"{source}: expected {expect} but received a web page",
                hint="The endpoint may have been retired or moved behind a login.",
            )
        if expect == "json":
            try:
                json.loads(resp.content)
            except ValueError as exc:
                raise ImmunoError(
                    ErrorCode.NON_DATA_RESPONSE,
                    f"{source}: expected JSON but got: {sanitize_message(resp.text, 160)}",
                ) from exc

    # ---- streaming download ---------------------------------------------------------------

    async def download(
        self,
        url: str,
        dest: Path,
        *,
        source: str,
        max_bytes: int = 200 * 1024 * 1024,
        expect_prefix: bytes | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> tuple[Path, str, int]:
        """Stream a file to ``dest`` and return (path, sha256, size). Written atomically."""
        if self.settings.offline:
            raise ImmunoError(ErrorCode.NOT_CONFIGURED, "offline mode is on; no network calls")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        digest, size, first = hashlib.sha256(), 0, b""
        try:
            async with self._client.stream("GET", url, headers=dict(headers or {})) as r:
                if r.status_code != 200:
                    raise ImmunoError(
                        ErrorCode.NOT_FOUND if r.status_code == 404 else ErrorCode.UPSTREAM_ERROR,
                        f"{source}: download failed with HTTP {r.status_code}",
                    )
                with tmp.open("wb") as fh:
                    async for chunk in r.aiter_bytes():
                        if not first:
                            first = chunk[:512]
                            if looks_like_html(first) or (
                                expect_prefix and not first.lstrip().startswith(expect_prefix)
                            ):
                                raise ImmunoError(
                                    ErrorCode.NON_DATA_RESPONSE,
                                    f"{source}: download is not the expected file type",
                                )
                        size += len(chunk)
                        if size > max_bytes:
                            raise ImmunoError(ErrorCode.TOO_LARGE, f"{source}: file exceeds limit")
                        digest.update(chunk)
                        fh.write(chunk)
            tmp.replace(dest)
        except httpx.HTTPError as exc:
            raise ImmunoError(
                ErrorCode.UPSTREAM_ERROR, f"{source}: download error ({type(exc).__name__})"
            ) from exc
        finally:
            tmp.unlink(missing_ok=True)
        return dest, digest.hexdigest(), size


def _retry_after(raw: httpx.Response) -> float | None:
    value = raw.headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value.strip()))
    except ValueError:
        return None

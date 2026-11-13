from __future__ import annotations

import hashlib

import httpx
import pytest

from immuno_mcp.core.errors import ErrorCode, ImmunoError
from immuno_mcp.core.http import RateLimiter, RetryPolicy, require_columns

URL = "https://api.example.org/v1/thing"


def json_ok(_: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"hello": "world"})


async def test_get_json_returns_parsed_body(make_http):
    http = make_http(json_ok)
    data, resp = await http.get_json("example", URL)
    assert data == {"hello": "world"} and resp.status == 200 and len(resp.sha256) == 64


async def test_web_page_with_status_200_is_an_error_not_data(make_http):
    page = "<!doctype html><html><body>Portal</body></html>"
    http = make_http(
        lambda r: httpx.Response(200, text=page, headers={"content-type": "text/html"})
    )
    with pytest.raises(ImmunoError) as err:
        await http.get_json("example", URL)
    assert err.value.code == ErrorCode.NON_DATA_RESPONSE


async def test_redirect_to_another_site_is_reported_as_moved(make_http):
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.host == "api.example.org":
            return httpx.Response(301, headers={"location": "https://new.example.com/api/thing"})
        return httpx.Response(
            200, text="<html>new site</html>", headers={"content-type": "text/html"}
        )

    http = make_http(handler)
    with pytest.raises(ImmunoError) as err:
        await http.get_json("example", URL)
    assert err.value.code == ErrorCode.UPSTREAM_MOVED and "new.example.com" in err.value.message


async def test_json_mislabelled_as_html_is_still_accepted(make_http):
    http = make_http(
        lambda r: httpx.Response(
            200, content=b'{"ok": true}', headers={"content-type": "text/html"}
        )
    )
    data, _ = await http.get_json("example", URL)
    assert data == {"ok": True}


async def test_text_error_served_with_200_is_caught_by_validator(make_http):
    http = make_http(lambda r: httpx.Response(200, text="Invalid allele name HLA-Z*99:99 found.\n"))
    with pytest.raises(ImmunoError) as err:
        await http.request(
            "POST",
            URL,
            source="iedb",
            expect="tsv",
            validate=lambda r: require_columns(r, ["allele"]),
        )
    assert err.value.code == ErrorCode.NON_DATA_RESPONSE and "Invalid allele" in err.value.message


async def test_valid_table_passes_the_validator(make_http):
    http = make_http(lambda r: httpx.Response(200, text="allele\tscore\nHLA-A*02:01\t0.8\n"))
    resp = await http.request(
        "GET", URL, source="iedb", expect="tsv", validate=lambda r: require_columns(r, ["allele"])
    )
    assert resp.text.startswith("allele")


async def test_retries_transient_failures_with_backoff(make_http):
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503) if calls["n"] < 3 else httpx.Response(200, json={"ok": 1})

    http = make_http(handler)
    data, _ = await http.get_json("example", URL)
    assert data == {"ok": 1} and calls["n"] == 3 and len(http.sleeps) == 2


async def test_retry_after_header_is_honoured(make_http):
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return (
            httpx.Response(429, headers={"retry-after": "2"})
            if calls["n"] == 1
            else httpx.Response(200, json={})
        )

    http = make_http(handler)
    await http.get_json("example", URL)
    assert http.sleeps == [2.0]


async def test_gives_up_after_max_attempts(make_http):
    http = make_http(lambda r: httpx.Response(503, text="down"))
    with pytest.raises(ImmunoError) as err:
        await http.get_json("example", URL)
    assert err.value.code == ErrorCode.UPSTREAM_ERROR and err.value.retryable


async def test_post_is_not_retried_after_a_server_error(make_http):
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500, text="boom")

    http = make_http(handler)
    with pytest.raises(ImmunoError):
        await http.post_json("example", URL, json_body={"x": 1})
    assert calls["n"] == 1  # a paid job must not be submitted twice


async def test_post_can_opt_in_to_retries_when_it_carries_an_idempotency_key(make_http):
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(502) if calls["n"] == 1 else httpx.Response(200, json={"id": "j1"})

    http = make_http(handler)
    data, _ = await http.post_json("example", URL, json_body={}, retry=RetryPolicy(retry_post=True))
    assert data == {"id": "j1"} and calls["n"] == 2


async def test_post_429_is_retried_because_nothing_was_processed(make_http):
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return (
            httpx.Response(429, headers={"retry-after": "1"})
            if calls["n"] == 1
            else httpx.Response(200, json={})
        )

    http = make_http(handler)
    await http.post_json("example", URL, json_body={})
    assert calls["n"] == 2


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (404, ErrorCode.NOT_FOUND),
        (422, ErrorCode.INVALID_INPUT),
        (400, ErrorCode.INVALID_INPUT),
        (401, ErrorCode.NEEDS_KEY),
        (403, ErrorCode.UPSTREAM_ERROR),
    ],
)
async def test_status_codes_map_to_typed_errors(make_http, status, code):
    http = make_http(lambda r: httpx.Response(status, text="detail: expected one of FV, FAB"))
    with pytest.raises(ImmunoError) as err:
        await http.get_json("example", URL)
    assert err.value.code == code


async def test_validation_errors_keep_the_servers_hint_for_the_model(make_http):
    http = make_http(
        lambda r: httpx.Response(422, json={"detail": "Input should be 'FV', 'FAB' or 'SD-H'"})
    )
    with pytest.raises(ImmunoError) as err:
        await http.get_json("example", URL)
    assert "SD-H" in err.value.message


async def test_good_responses_are_cached(make_http):
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"n": calls["n"]})

    http = make_http(handler, cache=True)
    first, r1 = await http.get_json("example", URL, params={"q": "a"}, ttl=60)
    second, r2 = await http.get_json("example", URL, params={"q": "a"}, ttl=60)
    other, _ = await http.get_json("example", URL, params={"q": "b"}, ttl=60)
    assert first == second == {"n": 1} and not r1.cached and r2.cached
    assert other == {"n": 2} and calls["n"] == 2


async def test_failed_responses_are_never_cached(make_http):
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(404)

    http = make_http(handler, cache=True)
    for _ in range(2):
        with pytest.raises(ImmunoError):
            await http.get_json("example", URL, ttl=60)
    assert calls["n"] == 2


async def test_offline_mode_blocks_the_network(make_http):
    http = make_http(json_ok, offline=True)
    with pytest.raises(ImmunoError) as err:
        await http.get_json("example", URL)
    assert err.value.code == ErrorCode.NOT_CONFIGURED


async def test_oversized_responses_are_refused(make_http):
    http = make_http(
        lambda r: httpx.Response(200, content=b"x" * 5000, headers={"content-type": "text/plain"})
    )
    with pytest.raises(ImmunoError) as err:
        await http.request("GET", URL, source="example", expect="text", max_bytes=1000)
    assert err.value.code == ErrorCode.TOO_LARGE


async def test_download_writes_file_atomically_and_hashes_it(make_http, tmp_path):
    body = b"data_1ABC\n#\nloop_\n"
    http = make_http(lambda r: httpx.Response(200, content=body))
    path, digest, size = await http.download(
        URL, tmp_path / "x" / "1abc.cif", source="rcsb", expect_prefix=b"data_"
    )
    assert path.read_bytes() == body and size == len(body)
    assert digest == hashlib.sha256(body).hexdigest()
    assert not list((tmp_path / "x").glob("*.part"))


async def test_download_rejects_a_web_page(make_http, tmp_path):
    http = make_http(lambda r: httpx.Response(200, content=b"<!DOCTYPE html><html></html>"))
    with pytest.raises(ImmunoError) as err:
        await http.download(URL, tmp_path / "f.cif", source="rcsb")
    assert err.value.code == ErrorCode.NON_DATA_RESPONSE
    assert not (tmp_path / "f.cif").exists()


async def test_rate_limiter_spaces_calls():
    now = {"t": 0.0}
    slept: list[float] = []

    async def fake_sleep(s: float) -> None:
        slept.append(s)
        now["t"] += s

    limiter = RateLimiter(clock=lambda: now["t"], sleep=fake_sleep)
    for _ in range(3):
        await limiter.wait("ncbi", rps=2.0)
    assert slept == pytest.approx([0.5, 0.5])
    await limiter.wait("other-source", rps=2.0)  # a different key is independent
    assert len(slept) == 2

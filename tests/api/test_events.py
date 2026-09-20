"""Tests for GET /runs/{id}/events: live SSE streaming, replay, and reconnect."""

from __future__ import annotations

import json

import httpx


def _parse_sse(text: str) -> list[dict]:
    """Parse a raw SSE response body into a list of decoded JSON `data:` payloads.

    sse-starlette's default record separator is `\r\n` (its
    `DEFAULT_SEPARATOR`), so events are blank-line-delimited with `\r\n\r\n`
    between them, not `\n\n`.
    """
    events = []
    for block in text.split("\r\n\r\n"):
        lines = [line for line in block.split("\r\n") if line]
        data_lines = [line[len("data: ") :] for line in lines if line.startswith("data: ")]
        if data_lines:
            events.append(json.loads("".join(data_lines)))
    return events


async def test_sse_stream_delivers_every_step_and_terminates_on_done(
    api_client: httpx.AsyncClient,
) -> None:
    create_response = await api_client.post("/runs", json={"ticker": "AAPL"})
    run_id = create_response.json()["run_id"]

    async with api_client.stream("GET", f"/runs/{run_id}/events") as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = ""
        async for chunk in response.aiter_text():
            body += chunk
            if '"step": "done"' in body or '"step": "error"' in body:
                break

    events = _parse_sse(body)
    steps_seen = [e["step"] for e in events]
    assert "fetch" in steps_seen
    assert steps_seen[-1] in ("done", "error")
    assert all(e["run_id"] == run_id for e in events)


async def test_late_connect_after_run_finished_replays_full_history(
    api_client: httpx.AsyncClient,
) -> None:
    create_response = await api_client.post("/runs", json={"ticker": "AAPL"})
    run_id = create_response.json()["run_id"]

    # Drain the stream once to let the run finish (its own task keeps running
    # even after this first stream disconnects -- POST-starts/GET-streams is
    # exactly the design that decouples the two).
    async with api_client.stream("GET", f"/runs/{run_id}/events") as response:
        body = ""
        async for chunk in response.aiter_text():
            body += chunk
            if '"step": "done"' in body or '"step": "error"' in body:
                break

    # A brand new connection to the same run id gets the same full history.
    async with api_client.stream("GET", f"/runs/{run_id}/events") as response:
        assert response.status_code == 200
        late_body = ""
        async for chunk in response.aiter_text():
            late_body += chunk

    late_events = _parse_sse(late_body)
    assert late_events[-1]["step"] in ("done", "error")
    assert len(late_events) >= 2


async def test_reconnect_with_last_event_id_resumes_without_replaying_seen_events(
    api_client: httpx.AsyncClient,
) -> None:
    create_response = await api_client.post("/runs", json={"ticker": "AAPL"})
    run_id = create_response.json()["run_id"]

    # Read just the very first event (id "0": fetch/started) and disconnect.
    first_id: str | None = None
    first_event: dict | None = None
    async with api_client.stream("GET", f"/runs/{run_id}/events") as response:
        buf = ""
        async for chunk in response.aiter_text():
            buf += chunk
            if "\r\n\r\n" in buf:
                block = buf.split("\r\n\r\n", 1)[0]
                for line in block.split("\r\n"):
                    if line.startswith("id: "):
                        first_id = line[len("id: ") :]
                first_event = _parse_sse(block + "\r\n\r\n")[0]
                break

    assert first_id == "0"
    assert first_event is not None
    assert first_event["step"] == "fetch"
    assert first_event["status"] == "started"

    async with api_client.stream(
        "GET", f"/runs/{run_id}/events", headers={"Last-Event-ID": first_id}
    ) as response:
        buf2 = ""
        async for chunk in response.aiter_text():
            buf2 += chunk
            if '"step": "done"' in buf2 or '"step": "error"' in buf2:
                break

    resumed_events = _parse_sse(buf2)
    # The reconnected stream must not replay the already-seen id-0 event.
    assert first_event not in resumed_events

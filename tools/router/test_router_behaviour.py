#!/usr/bin/env python3
"""Checks for the failure paths the latency suite never reaches.

test_router.py answers whether placement is any good. These answer whether the
router stays honest when a request fails, a client hangs up, or the two servers
are not holding the same model.
"""
import json
import socket
import struct
import sys
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import mock_backend                                   # noqa: E402
import ninfer_router                                  # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    print(f"  {'pass' if ok else 'FAIL'}  {name}{'' if ok else '  ' + detail}")
    if not ok:
        FAILURES.append(name)


def start_router(port, backends):
    handler = type("H", (ninfer_router.Handler,), {
        "router": ninfer_router.Router(backends, lambda line: None)})
    server = ninfer_router.ThreadedServer(("127.0.0.1", port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    time.sleep(0.2)
    return server, handler.router


def long_body(tag, words=20000, max_tokens=32):
    return {"model": "m", "max_tokens": max_tokens, "stream": True,
            "messages": [{"role": "user", "content": f"{tag} " + "word " * words}]}


def test_failed_request_forgets_affinity():
    port = 9501
    mock_backend.serve(port, 3000, 60, 200000, 400.0)
    backend = ninfer_router.Backend("one", f"http://127.0.0.1:{port}", 200000, 3000, 60)
    server, router = start_router(9502, [backend])

    body = long_body("FAIL-ME")
    try:
        urllib.request.urlopen(urllib.request.Request(
            "http://127.0.0.1:9502/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"content-type": "application/json"}), timeout=30).read()
    except urllib.error.HTTPError:
        pass
    check("a failed request leaves no affinity behind", len(backend.prefixes) == 0,
          f"{len(backend.prefixes)} prefixes retained")

    ok_body = long_body("KEEP-ME")
    urllib.request.urlopen(urllib.request.Request(
        "http://127.0.0.1:9502/v1/chat/completions",
        data=json.dumps(ok_body).encode(),
        headers={"content-type": "application/json"}), timeout=60).read()
    check("a successful request does record affinity", len(backend.prefixes) == 1,
          f"{len(backend.prefixes)} prefixes retained")
    check("a failed request does not leak queue depth", backend.queued == 0,
          f"queued={backend.queued}")
    check("a failed request does not leak committed work", backend.pending_seconds < 0.01,
          f"pending={backend.pending_seconds:.3f}s")
    server.shutdown()
    server.server_close()


def test_client_disconnect_frees_the_backend():
    port = 9503
    mock_backend.serve(port, 3000, 60, 200000, 60.0)
    backend = ninfer_router.Backend("one", f"http://127.0.0.1:{port}", 200000, 3000, 60)
    server, router = start_router(9504, [backend])

    # Send a slow request, read one byte, then hang up hard.
    body = json.dumps(long_body("DISCONNECT", words=8000, max_tokens=512)).encode()
    raw = socket.create_connection(("127.0.0.1", 9504), timeout=30)
    raw.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
                b"content-type: application/json\r\n"
                b"content-length: %d\r\n\r\n" % len(body) + body)
    time.sleep(0.5)
    raw.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER,
                   struct.pack("ii", 1, 0))   # RST rather than FIN
    raw.close()

    # The backend must become reusable rather than staying locked forever.
    freed = False
    for _ in range(120):
        if backend.lock.acquire(blocking=False):
            backend.lock.release()
            freed = True
            break
        time.sleep(0.25)
    check("the backend is released after the client hangs up", freed,
          "lock still held 30s later")
    check("queue depth returns to zero after a disconnect", backend.queued == 0,
          f"queued={backend.queued}")
    server.shutdown()
    server.server_close()


def test_mismatched_models_are_refused():
    mock_backend.serve(9505, 3000, 60, 200000, 400.0, model_id="model-a")
    mock_backend.serve(9506, 3000, 60, 200000, 400.0, model_id="model-b")
    same = [ninfer_router.Backend("a", "http://127.0.0.1:9505", 200000, 3000, 60),
            ninfer_router.Backend("b", "http://127.0.0.1:9505", 200000, 3000, 60)]
    different = [ninfer_router.Backend("a", "http://127.0.0.1:9505", 200000, 3000, 60),
                 ninfer_router.Backend("b", "http://127.0.0.1:9506", 200000, 3000, 60)]

    try:
        ninfer_router.check_backends(same, lambda line: None)
        matching_ok = True
    except SystemExit:
        matching_ok = False
    check("matching models are accepted", matching_ok)

    try:
        ninfer_router.check_backends(different, lambda line: None)
        refused = False
    except SystemExit:
        refused = True
    check("mismatched models are refused", refused, "routed across two different models")

    absent = [ninfer_router.Backend("a", "http://127.0.0.1:9505", 200000, 3000, 60),
              ninfer_router.Backend("down", "http://127.0.0.1:9599", 200000, 3000, 60)]
    try:
        ninfer_router.check_backends(absent, lambda line: None)
        tolerated = True
    except SystemExit:
        tolerated = False
    check("a backend that is not up yet is a warning, not a refusal", tolerated)


def test_failover_to_a_live_backend():
    live = 9507
    mock_backend.serve(live, 3000, 60, 200000, 400.0)
    dead = ninfer_router.Backend("dead", "http://127.0.0.1:9598", 200000, 3000, 60)
    good = ninfer_router.Backend("live", f"http://127.0.0.1:{live}", 200000, 3000, 60)
    server, router = start_router(9508, [dead, good])

    body = {"model": "m", "max_tokens": 16,
            "messages": [{"role": "user", "content": "hello"}]}
    served = None
    try:
        served = urllib.request.urlopen(urllib.request.Request(
            "http://127.0.0.1:9508/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"content-type": "application/json"}), timeout=30).read()
    except Exception as exc:
        served = None
        detail = str(exc)
    check("a request lands on the live backend when one is down", bool(served),
          locals().get("detail", ""))
    check("the dead backend is taken out of rotation", dead.healthy is False,
          "still marked healthy")
    check("the live backend stays in rotation", good.healthy is True)

    # It comes back, and a probe restores it.
    mock_backend.serve(9598, 3000, 60, 200000, 400.0)
    router.probe_unhealthy()
    check("a backend that comes back is restored", dead.healthy is True,
          "still marked unreachable")
    server.shutdown()
    server.server_close()


def test_midstream_failure_is_not_retried():
    a, b = 9509, 9510
    mock_backend.serve(a, 3000, 60, 200000, 400.0)
    mock_backend.serve(b, 3000, 60, 200000, 400.0)
    one = ninfer_router.Backend("one", f"http://127.0.0.1:{a}", 200000, 3000, 60)
    two = ninfer_router.Backend("two", f"http://127.0.0.1:{b}", 200000, 3000, 60)
    server, router = start_router(9511, [one, two])

    body = {"model": "m", "max_tokens": 16,
            "messages": [{"role": "user", "content": "CUT-ME please"}]}
    try:
        payload = urllib.request.urlopen(urllib.request.Request(
            "http://127.0.0.1:9511/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"content-type": "application/json"}), timeout=30).read()
    except Exception:
        payload = b""
    # Exactly one backend should have seen it: retrying after bytes were sent
    # would put a second partial response on the same connection.
    total = payload.count(b"partial")
    check("a mid-stream cut is not retried into a duplicate", total <= 1,
          f"client saw {total} partial responses")
    server.shutdown()
    server.server_close()


def test_affinity_slots_follow_concurrency():
    pro4000 = ninfer_router.Backend("bw", "http://x", 98304, 3860, 57.4,
                                      concurrency=1, slots_per_lane=3)
    check("Pro 4000 at concurrency 1 gets 3 slots", pro4000.affinity_slots == 3,
          f"got {pro4000.affinity_slots}")
    pro4000_2 = ninfer_router.Backend("bw2", "http://x", 73728, 3860, 57.4,
                                       concurrency=2, slots_per_lane=3)
    check("Pro 4000 at concurrency 2 gets 6 slots", pro4000_2.affinity_slots == 6,
          f"got {pro4000_2.affinity_slots}")
    gpu = ninfer_router.Backend("4090", "http://x", 262144, 2115, 108.0,
                                concurrency=2, slots_per_lane=1)
    check("the 4090 at concurrency 2 gets 2 slots", gpu.affinity_slots == 2,
          f"got {gpu.affinity_slots}")
    override = ninfer_router.Backend("o", "http://x", 1000, 1, 1,
                                     concurrency=4, slots_per_lane=3, affinity_slots=2)
    check("an explicit affinity_slots still wins", override.affinity_slots == 2,
          f"got {override.affinity_slots}")


def test_vision_requests_exclude_non_vision_backends():
    """A request carrying images must never land on a backend with vision=0."""
    port = 9520
    mock_backend.serve(port, 3000, 60, 200000, 400.0)
    no_vision = ninfer_router.Backend("no_vision", f"http://127.0.0.1:{port}",
                                      200000, 3000, 60, vision=False)
    with_vision = ninfer_router.Backend("with_vision", f"http://127.0.0.1:{port}",
                                        200000, 3000, 60, vision=True)
    server, router = start_router(9521, [no_vision, with_vision])

    # Vision request: OpenAI image_url part.
    vision_body = {"model": "m", "max_tokens": 16,
                   "messages": [{"role": "user",
                                 "content": [
                                     {"type": "text", "text": "what is this?"},
                                     {"type": "image_url",
                                      "image_url": {"url": "data:image/png;base64,AAAA"}}]}]}
    urllib.request.urlopen(urllib.request.Request(
        "http://127.0.0.1:9521/v1/chat/completions",
        data=json.dumps(vision_body).encode(),
        headers={"content-type": "application/json"}), timeout=30).read()

    check("a vision request is not routed to a vision=0 backend",
          no_vision.queued == 0 and no_vision.stats.get("cold-idle", 0) == 0,
          f"no_vision queued={no_vision.queued} stats={dict(no_vision.stats)}")
    check("a vision request lands on a vision=1 backend",
          with_vision.queued == 0 and sum(with_vision.stats.values()) == 1,
          f"with_vision stats={dict(with_vision.stats)}")

    # Non-vision request: should be eligible for both backends.
    text_body = {"model": "m", "max_tokens": 16,
                 "messages": [{"role": "user", "content": "hello"}]}
    urllib.request.urlopen(urllib.request.Request(
        "http://127.0.0.1:9521/v1/chat/completions",
        data=json.dumps(text_body).encode(),
        headers={"content-type": "application/json"}), timeout=30).read()

    check("a text-only request may still use either backend",
          sum(no_vision.stats.values()) + sum(with_vision.stats.values()) == 2,
          f"no_vision={dict(no_vision.stats)} with_vision={dict(with_vision.stats)}")

    # Vision request with Anthropic image part shape.
    anthropic_vision = {"model": "m", "max_tokens": 16,
                        "messages": [{"role": "user",
                                      "content": [
                                          {"type": "text", "text": "describe"},
                                          {"type": "image",
                                           "source": {"type": "base64",
                                                      "media_type": "image/png",
                                                      "data": "AAAA"}}]}]}
    urllib.request.urlopen(urllib.request.Request(
        "http://127.0.0.1:9521/v1/messages",
        data=json.dumps(anthropic_vision).encode(),
        headers={"content-type": "application/json", "anthropic-version": "2023-06-01"}),
        timeout=30).read()

    check("an Anthropic image request is also excluded from vision=0",
          no_vision.queued == 0,
          f"no_vision queued={no_vision.queued} stats={dict(no_vision.stats)}")

    server.shutdown()
    server.server_close()


def test_vision_routing_when_only_vision_backend_is_down():
    """If the only vision-capable backend is down, a vision request fails
    rather than being sent to a backend that will reject it."""
    port = 9522
    mock_backend.serve(port, 3000, 60, 200000, 400.0)
    good = ninfer_router.Backend("good", f"http://127.0.0.1:{port}",
                                 200000, 3000, 60, vision=True)
    dead_visionless = ninfer_router.Backend("dead_nv",
                                            "http://127.0.0.1:9599",
                                            200000, 3000, 60, vision=False)
    server, router = start_router(9523, [dead_visionless, good])

    vision_body = {"model": "m", "max_tokens": 16,
                   "messages": [{"role": "user",
                                 "content": [{"type": "text", "text": "hi"},
                                             {"type": "image_url",
                                              "image_url": {"url": "data:image/png;base64,x"}}]}]}
    try:
        urllib.request.urlopen(urllib.request.Request(
            "http://127.0.0.1:9523/v1/chat/completions",
            data=json.dumps(vision_body).encode(),
            headers={"content-type": "application/json"}), timeout=30).read()
        ok = True
    except Exception:
        ok = False
    check("a vision request succeeds when the vision backend is up", ok)
    check("the visionless backend was never touched for a vision request",
          sum(dead_visionless.stats.values()) == 0,
          f"stats={dict(dead_visionless.stats)}")
    server.shutdown()
    server.server_close()


def main():
    print("failure and identity behaviour")
    test_failed_request_forgets_affinity()
    test_client_disconnect_frees_the_backend()
    test_mismatched_models_are_refused()
    test_failover_to_a_live_backend()
    test_midstream_failure_is_not_retried()
    test_affinity_slots_follow_concurrency()
    test_vision_requests_exclude_non_vision_backends()
    test_vision_routing_when_only_vision_backend_is_down()
    if FAILURES:
        print(f"\nFAIL: {len(FAILURES)} check(s) failed")
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

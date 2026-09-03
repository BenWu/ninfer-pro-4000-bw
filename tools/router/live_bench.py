#!/usr/bin/env python3
"""Compare placement policies against two real ninfer servers.

The mock suite in test_router.py checks that the router makes the right choices.
This checks that those choices are worth making on the actual hardware, which is
the only claim that matters. Run with both cards serving.

    python3 tools/router/live_bench.py \
        --pro4000 http://127.0.0.1:8080 --rtx4090 http://127.0.0.1:8081
"""
import argparse
import json
import os
import statistics
import sys
import threading
import time
import urllib.request

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import ninfer_router                                  # noqa: E402


def call(url, body, path="/v1/messages"):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json",
                                          "accept": "text/event-stream",
                                          "anthropic-version": "2023-06-01"})
    start = time.perf_counter()
    ttft = None
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or ttft is not None:
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":          # OpenAI terminates the stream this way
                continue
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "content_block_delta":
                ttft = time.perf_counter() - start
            else:
                delta = ((event.get("choices") or [{}])[0]).get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content"):
                    ttft = time.perf_counter() - start
    total = time.perf_counter() - start
    return ttft if ttft is not None else total, total


def run_trace(targets, trace, policy, stagger=0.2, path="/v1/messages"):
    results = [None] * len(trace)
    counter = {"i": 0}
    lock = threading.Lock()

    def worker(index, body):
        with lock:
            slot = counter["i"]
            counter["i"] += 1
        results[index] = call(policy(targets, slot), body, path)

    threads = []
    for index, body in enumerate(trace):
        thread = threading.Thread(target=worker, args=(index, body))
        threads.append(thread)
        thread.start()
        time.sleep(stagger)
    for thread in threads:
        thread.join()
    return results


def report(name, results):
    ttfts = [r[0] for r in results]
    totals = [r[1] for r in results]
    return (f"  {name:14s} ttft mean={statistics.mean(ttfts):7.3f}s "
            f"median={statistics.median(ttfts):7.3f}s max={max(ttfts):7.3f}s   "
            f"total mean={statistics.mean(totals):7.3f}s")


# Unique per invocation, and per policy within one. Sharing a nonce across the
# four policies lets each one run warmer than the last, because the earlier
# policies leave the prefixes cached, which silently flatters whichever policy
# runs last by turning its cold prefills into hits.
RUN_ID = os.urandom(4).hex()


def load(path, tag, tail, endpoint="/v1/messages", model="qwen3.8-27b", nonce=""):
    tag = f"{RUN_ID}-{nonce}-{tag}"
    messages = [dict(m) for m in json.load(open(path))]
    body = {"model": model, "max_tokens": 48, "temperature": 0, "stream": True}
    if endpoint == "/v1/chat/completions":
        # OpenAI keeps the system turn in the messages array.
        first_user = next(i for i, m in enumerate(messages) if m["role"] != "system")
        messages[first_user]["content"] = f"{tag} " + messages[first_user]["content"] + tail
        body["messages"] = messages
        return body
    system = messages[0]["content"] if messages[0]["role"] == "system" else None
    rest = [m for m in messages if m["role"] != "system"]
    rest[0]["content"] = f"{tag} " + rest[0]["content"] + tail
    body["messages"] = rest
    if system:
        body["system"] = system
    return body


def scenario_head_of_line(args, nonce=""):
    trace = [load(args.long_prompt, "HOL", "\n\nSummarize in one sentence.",
                  args.endpoint, nonce=nonce)]
    trace += [{"model": "qwen3.8-27b", "max_tokens": 48, "temperature": 0, "stream": True,
               "messages": [{"role": "user", "content": f"What is {i} times 37?"}]}
              for i in range(6)]
    return "one cold 32k prefill, then 6 short calls", trace


def scenario_affinity(args, nonce=""):
    order = ["A", "A", "B", "A", "B", "B", "A", "B"]
    trace = []
    for tag in order:
        trace.append(load(args.mid_prompt, f"DOC-{tag}",
                          "\n\nSummarize in one sentence.", args.endpoint, nonce=nonce))
    return "2 documents, 8 bunched turns", trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pro4000", required=True)
    ap.add_argument("--rtx4090", required=True)
    ap.add_argument("--router-port", type=int, default=8099)
    ap.add_argument("--long-prompt", default="models/long_prompt_32tok.json")
    ap.add_argument("--mid-prompt", default="models/retrieval_spot.json")
    ap.add_argument("--endpoint", default="/v1/messages",
                    choices=["/v1/messages", "/v1/chat/completions"])
    args = ap.parse_args()

    urls = [args.pro4000, args.rtx4090]
    router_url = f"http://127.0.0.1:{args.router_port}"

    for build in (scenario_head_of_line, scenario_affinity):
        name, _ = build(args)
        print(f"\n{name}")
        ep = args.endpoint

        # Each policy gets its own nonce, so all four face equally cold caches.
        _, trace = build(args, "single-bw")
        print(report("pro4000", run_trace([urls[0]], trace, lambda t, i: t[0], path=ep)))
        _, trace = build(args, "single-gpu")
        print(report("rtx4090", run_trace([urls[1]], trace, lambda t, i: t[0], path=ep)))
        _, trace = build(args, "roundrobin")
        print(report("round-robin",
                     run_trace(urls, trace, lambda t, i: t[i % len(t)], path=ep)))

        backends = [
            ninfer_router.Backend("pro4000", urls[0], 98304, 3860, 57.4,
                                  attention_s_per_token2=2.493e-9,
                                  concurrency=1, slots_per_lane=3),
            # The 4090 fork has no context cache: reuse is per lane, so it
            # holds one context per --max-concurrency. It runs 2.
            ninfer_router.Backend("rtx4090", urls[1], 262144, 2115, 108.0,
                                  attention_s_per_token2=1.619e-9,
                                  concurrency=2, slots_per_lane=1),
        ]
        handler = type("H", (ninfer_router.Handler,), {
            "router": ninfer_router.Router(backends, lambda line: None)})
        server = ninfer_router.ThreadedServer(("127.0.0.1", args.router_port), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        time.sleep(0.3)
        _, trace = build(args, "router")
        print(report("router", run_trace([router_url], trace, lambda t, i: t[0], path=ep)))
        placements = {n: i["routed"] for n, i in handler.router.snapshot().items()}
        print(f"  placements: {placements}")
        server.shutdown()
        server.server_close()
        time.sleep(0.3)


if __name__ == "__main__":
    main()

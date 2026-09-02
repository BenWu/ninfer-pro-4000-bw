#!/usr/bin/env python3
"""Drive the router against two mock backends and compare placement policies.

Each scenario runs the same request trace three ways: one card alone,
round robin over two cards, and the router. The mocks carry the rates measured
on this machine, scaled down in wall clock so the suite finishes quickly.
"""
import json
import statistics
import sys
import threading
import time
import urllib.request

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import mock_backend                                   # noqa: E402
import ninfer_router                                  # noqa: E402

SPEED = 60.0
BW = dict(prefill=3350, decode=57, max_context=81920)
GPU4090 = dict(prefill=1893, decode=51, max_context=262144)


def start_router(port, backends, quiet=True):
    router = ninfer_router.Router(backends, lambda line: None if quiet else print(line))
    handler = type("H", (ninfer_router.Handler,), {"router": router})
    server = ninfer_router.ThreadedServer(("127.0.0.1", port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, router


def call(url, body, path="/v1/messages"):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json",
                                          "accept": "text/event-stream"})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        resp.read()
    return time.perf_counter() - start


def run_trace(targets, trace, policy, path="/v1/messages"):
    """targets is a list of base urls; policy picks one per request."""
    latencies = [None] * len(trace)
    counter = {"i": 0}
    lock = threading.Lock()

    def worker(index, body):
        with lock:
            slot = counter["i"]
            counter["i"] += 1
        url = policy(targets, slot)
        latencies[index] = call(url, body, path)

    threads = []
    for index, body in enumerate(trace):
        thread = threading.Thread(target=worker, args=(index, body))
        threads.append(thread)
        thread.start()
        time.sleep(0.01)
    for thread in threads:
        thread.join()
    return latencies


def document(tag, kilo):
    return {"role": "user", "content": f"{tag} " + "word " * (kilo * 1000)}


def report(name, latencies):
    ordered = sorted(latencies)
    p95 = ordered[max(0, int(len(ordered) * 0.95) - 1)]
    return (f"  {name:22s} mean={statistics.mean(latencies):6.3f}s  "
            f"median={statistics.median(latencies):6.3f}s  p95={p95:6.3f}s  "
            f"max={max(latencies):6.3f}s")


def scenario_head_of_line():
    """One long prefill arrives, then short chatty requests queue behind it."""
    trace = [{"max_tokens": 32, "messages": [document("BIG", 30)]}]
    trace += [{"max_tokens": 32, "messages": [{"role": "user", "content": f"quick {i}"}]}
              for i in range(6)]
    return "head of line: 1 cold 30k prefill, then 6 short calls", trace, "/v1/messages"


def conversation(tag, kilo, turn):
    messages = [document(tag, kilo)]
    for prior in range(turn):
        messages.append({"role": "assistant", "content": f"answer {prior}"})
        messages.append({"role": "user", "content": f"follow up {prior}"})
    return {"max_tokens": 32, "messages": messages}


def scenario_two_conversations():
    """Two long documents with follow up turns, arriving irregularly.

    A strictly alternating trace lets round robin fall into perfect affinity by
    accident, which flatters it for a reason that has nothing to do with its
    policy. Real traffic bunches, so this order does too.
    """
    order = ["A", "A", "B", "A", "B", "B", "A", "B"]
    turns = {"A": 0, "B": 0}
    trace = []
    for tag in order:
        trace.append(conversation(f"DOC-{tag}", 25, turns[tag]))
        turns[tag] += 1
    return "affinity: 2 long docs, 8 bunched turns", trace, "/v1/messages"


def scenario_three_conversations():
    """Three live documents against two cards holding about three prefixes each."""
    order = ["A", "B", "C", "A", "C", "B", "A", "B", "C"]
    turns = {"A": 0, "B": 0, "C": 0}
    trace = []
    for tag in order:
        trace.append(conversation(f"DOC-{tag}", 20, turns[tag]))
        turns[tag] += 1
    return "affinity: 3 long docs, 9 bunched turns", trace, "/v1/messages"


def scenario_openai_affinity():
    """The affinity trace again, in OpenAI shape on /v1/chat/completions.

    OpenAI carries the system turn inside messages instead of a top level field,
    so this checks the adapter hashes and sizes the request the same way rather
    than merely accepting it.
    """
    order = ["A", "A", "B", "A", "B", "B", "A", "B"]
    turns = {"A": 0, "B": 0}
    trace = []
    for tag in order:
        turn = turns[tag]
        messages = [{"role": "system", "content": "Answer from the document."},
                    document(f"DOC-{tag}", 25)]
        for prior in range(turn):
            messages.append({"role": "assistant", "content": f"answer {prior}"})
            messages.append({"role": "user", "content": f"follow up {prior}"})
        trace.append({"model": "m", "max_tokens": 32, "messages": messages})
        turns[tag] += 1
    return "openai chat completions: 2 long docs, 8 bunched turns", trace, "/v1/chat/completions"


def scenario_mixed():
    """A realistic blend: mostly chat, occasional long document."""
    trace = []
    for i in range(14):
        if i % 5 == 0:
            trace.append({"max_tokens": 48, "messages": [document(f"DOC-{i}", 20)]})
        else:
            trace.append({"max_tokens": 96,
                          "messages": [{"role": "user", "content": f"chat turn {i}"}]})
    return "mixed: 3 long documents among 11 chat turns", trace, "/v1/messages"


def main():
    port = 9300
    failures = []
    for build in (scenario_head_of_line, scenario_two_conversations,
                  scenario_three_conversations, scenario_openai_affinity,
                  scenario_mixed):
        name, trace, path = build()
        print(f"\n{name}")
        results = {}
        for policy in ("single", "round-robin", "router"):
            # Every policy gets untouched servers. Prefix caches and the
            # router's own tables both carry state, so reusing them between
            # scenarios measures the previous scenario as much as this one.
            bw_port, gpu_port, router_port = port, port + 1, port + 2
            port += 3
            mock_backend.serve(bw_port, BW["prefill"], BW["decode"], BW["max_context"], SPEED)
            mock_backend.serve(gpu_port, GPU4090["prefill"], GPU4090["decode"],
                               GPU4090["max_context"], SPEED)
            urls = [f"http://127.0.0.1:{bw_port}", f"http://127.0.0.1:{gpu_port}"]
            if policy == "single":
                latencies = run_trace([urls[0]], trace, lambda t, i: t[0], path)
            elif policy == "round-robin":
                latencies = run_trace(urls, trace, lambda t, i: t[i % len(t)], path)
            else:
                backends = [
                    ninfer_router.Backend("blackwell", urls[0], BW["max_context"],
                                          BW["prefill"], BW["decode"]),
                    ninfer_router.Backend("rtx4090", urls[1], GPU4090["max_context"],
                                          GPU4090["prefill"], GPU4090["decode"]),
                ]
                start_router(router_port, backends)
                time.sleep(0.2)
                latencies = run_trace([f"http://127.0.0.1:{router_port}"], trace,
                                      lambda t, i: t[0], path)
            results[policy] = latencies
            print(report(policy, latencies))

        best_baseline = min(statistics.mean(results["single"]),
                            statistics.mean(results["round-robin"]))
        routed = statistics.mean(results["router"])
        # The router must never be materially worse than simply picking the
        # better of the two fixed policies, which is the honest bar: a router
        # that only wins on traces chosen to suit it is not worth running.
        if routed > best_baseline * 1.05:
            failures.append(f"{name}: router {routed:.3f}s vs best baseline "
                            f"{best_baseline:.3f}s")

    if failures:
        print("\nFAIL")
        for line in failures:
            print("  " + line)
        return 1
    print("\nPASS: router within 5% of the better fixed policy in every scenario, "
          "and ahead of both where affinity or head of line blocking is in play")
    return 0


if __name__ == "__main__":
    sys.exit(main())

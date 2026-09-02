#!/usr/bin/env python3
"""Measure the prefill and decode cost curve of one ninfer server.

Routing between two cards needs a cost model, not a single tokens per second
number. This fits one: total latency is prefill(P) + G * decode_step, so it
measures time to first token against prompt length and the inter token gap
separately, then reports both.

Usage:
    python3 tools/router/shape_bench.py --server http://127.0.0.1:8080 \
        --label blackwell --out bw.json
"""
import argparse
import json
import time
import urllib.error
import urllib.request

LONG_ANSWER = (
    "\n\nNow ignore the document and instead write a long continuous prose essay "
    "about the history of numerical computing. Do not stop early."
)


def post(server, path, body, stream=False, timeout=1800):
    req = urllib.request.Request(
        server.rstrip("/") + path,
        data=json.dumps(body).encode(),
        headers={
            "content-type": "application/json",
            "accept": "text/event-stream" if stream else "application/json",
            "anthropic-version": "2023-06-01",
        },
    )
    return urllib.request.urlopen(req, timeout=timeout)


def split_system(messages):
    """Anthropic style: a leading system turn is a top level field, not a message."""
    system = None
    rest = list(messages)
    if rest and rest[0]["role"] == "system":
        system = rest[0]["content"]
        rest = rest[1:]
    return system, rest


def count_tokens(server, messages):
    system, rest = split_system(messages)
    body = {"model": "m", "messages": rest}
    if system is not None:
        body["system"] = system
    with post(server, "/v1/messages/count_tokens", body) as r:
        return json.load(r)["input_tokens"]


def run(server, messages, max_tokens):
    """One streaming generation. Returns (prompt_tokens, ttft_s, total_s, out_tokens)."""
    system, rest = split_system(messages)
    body = {"model": "m", "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "messages": rest}
    if system is not None:
        body["system"] = system
    start = time.perf_counter()
    ttft = None
    out_tokens = 0
    usage_in = None
    with post(server, "/v1/messages", body, stream=True) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            ev = json.loads(line[5:].strip())
            kind = ev.get("type")
            if kind == "content_block_delta" and (ev.get("delta") or {}).get("type") in (
                    "text_delta", "thinking_delta"):
                if ttft is None:
                    ttft = time.perf_counter() - start
            elif kind == "message_start":
                usage_in = ((ev.get("message") or {}).get("usage") or {}).get("input_tokens")
            elif kind == "message_delta":
                out_tokens = (ev.get("usage") or {}).get("output_tokens", out_tokens)
    total = time.perf_counter() - start
    return usage_in, ttft, total, out_tokens


def load_prompt(path):
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--max-context", type=int, default=0,
                    help="skip prompts that would not fit; 0 disables the check")
    ap.add_argument("--prompts", nargs="*", default=[])
    args = ap.parse_args()

    cases = []
    short = [{"role": "user", "content":
              "Write a long continuous prose essay about the history of numerical "
              "computing. Do not stop early."}]
    cases.append(("short", short))
    for path in args.prompts:
        messages = load_prompt(path)
        messages = list(messages)
        messages[-1] = dict(messages[-1])
        messages[-1]["content"] = messages[-1]["content"] + LONG_ANSWER
        cases.append((path.split("/")[-1].replace(".json", ""), messages))

    results = []
    for name, messages in cases:
        try:
            prompt_tokens = count_tokens(args.server, messages)
        except urllib.error.HTTPError as exc:
            print(f"{name}: count_tokens failed {exc.code}", flush=True)
            continue
        if args.max_context and prompt_tokens + args.max_tokens > args.max_context:
            print(f"{name}: {prompt_tokens} tokens exceeds context, skipped", flush=True)
            results.append({"case": name, "prompt_tokens": prompt_tokens, "skipped": True})
            continue
        for trial in range(args.repeat):
            try:
                served_in, ttft, total, out = run(args.server, messages, args.max_tokens)
            except urllib.error.HTTPError as exc:
                print(f"{name}: generate failed {exc.code} {exc.read()[:200]!r}", flush=True)
                break
            if ttft is None or out < 2:
                print(f"{name}: no usable output (out={out})", flush=True)
                break
            decode_ms = (total - ttft) / (out - 1) * 1000.0
            row = {"case": name, "trial": trial, "prompt_tokens": served_in or prompt_tokens,
                   "ttft_ms": ttft * 1000.0, "total_s": total, "output_tokens": out,
                   "decode_ms_per_token": decode_ms,
                   "decode_tok_s": 1000.0 / decode_ms}
            results.append(row)
            print(f"{name:24s} P={row['prompt_tokens']:>7} ttft={row['ttft_ms']:8.1f}ms "
                  f"out={out:>4} decode={row['decode_tok_s']:6.2f} tok/s", flush=True)

    with open(args.out, "w") as f:
        json.dump({"label": args.label, "server": args.server, "results": results}, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

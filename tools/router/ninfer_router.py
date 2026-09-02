#!/usr/bin/env python3
"""Affinity-first router for two ninfer servers on one machine.

The measured facts this is built on, all from this machine (see README.md):

  - An exact prefix hit turns an 11 second prefill into 0.10 seconds, a 108x
    swing. Nothing else the router can decide comes close, so cache affinity
    outranks every other rule.
  - Servers run --max-concurrency 1, so a request queued behind a cold 32k
    prefill waits for all of it: a 0.6 second call measured 11.6 seconds. The
    queue therefore lives here, one in flight per backend, and never in the
    server where the router cannot see or reorder it.
  - The Blackwell holds 81920 tokens of context against the 4090's 262144, so
    long prompts are a placement constraint before they are a preference.

Shape only breaks ties between backends that are equally warm and equally idle.

Usage:
    python3 tools/router/ninfer_router.py --port 8090 \
        --backend blackwell=http://127.0.0.1:8080,max_context=81920,prefill=3860,decode=57.4,attn=2.493e-9 \
        --backend rtx4090=http://127.0.0.1:8081,max_context=262144,prefill=2115,decode=108.0,attn=1.619e-9
"""
import argparse
import collections
import hashlib
import json
import queue
import socketserver
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler

# Conservative: the corpora on this machine run 3.45 to 3.87 characters per
# token, so dividing by 3 overestimates and errs toward the larger context.
CHARS_PER_TOKEN = 3.0

# Roughly the measured retention of one card at --max-concurrency 1: two private
# continuations plus one shared prefix. Tracking more would promise hits the
# server cannot honour.
AFFINITY_SLOTS = 3

# Retention is a cost model decision inside the engine, not a fixed rule, so this
# is a measured floor rather than a policy. Cycling two prompts on the Blackwell
# with calibrated cost presets: 349 tokens never retained, 949 retained a third
# of the time, 1949 half, 3943 and above every time. Below this the router must
# not pin a request to a card for a hit the server will not give, because that
# only costs it balance. Re-measure after changing --max-concurrency or the
# context cost presets, both of which move the floor.
AFFINITY_MIN_TOKENS = 2048


class Backend:
    def __init__(self, name, url, max_context, prefill_tok_s, decode_tok_s,
                 affinity_min_tokens=AFFINITY_MIN_TOKENS, attention_s_per_token2=0.0):
        self.name = name
        self.url = url.rstrip("/")
        self.max_context = max_context
        self.prefill_tok_s = prefill_tok_s
        self.decode_tok_s = decode_tok_s
        # Prefill is not linear: measured on the Blackwell, a flat tokens per
        # second rate underestimates a 63k prompt by 28%, and the gap widens
        # with length. Attention over the prompt is the quadratic part.
        self.attention_s_per_token2 = attention_s_per_token2
        self.lock = threading.Lock()      # one in flight, matching --max-concurrency 1
        self.queued = 0                   # waiting plus running, for placement
        self.prefixes = collections.OrderedDict()   # prefix hash -> token estimate
        self.stats = collections.Counter()
        self.affinity_min_tokens = affinity_min_tokens
        # Estimated work already committed to this backend. Counting jobs and
        # multiplying by an average badly misprices a queue whose jobs differ by
        # more than an order of magnitude, which is exactly the traffic here:
        # a cold 30k prefill and a chat turn are not interchangeable units.
        self.pending_seconds = 0.0

    def remember(self, hashes):
        """Record what this backend will hold once the request completes.

        Only the full context is retained, not every intermediate prefix: the
        server keeps the continuation it just finished, and storing each turn
        separately would let a single multi turn request evict every other
        conversation from a table this small.
        """
        if not hashes:
            return
        digest, tokens = hashes[-1]
        if tokens < self.affinity_min_tokens:
            return
        self.prefixes.pop(digest, None)
        self.prefixes[digest] = tokens
        while len(self.prefixes) > AFFINITY_SLOTS:
            self.prefixes.popitem(last=False)

    def warm_tokens(self, hashes):
        """Tokens of the longest prefix of this request the backend still holds."""
        best = 0
        for digest, tokens in hashes:
            if digest in self.prefixes:
                best = max(best, tokens)
        return best

    def estimate_seconds(self, prompt_tokens, warm_tokens, output_tokens):
        """Seconds to prefill what is not cached, then generate.

        A warm prefix removes both the token work and the attention work below
        it, so the quadratic part is a difference of squares rather than a
        function of the recomputed count alone.
        """
        warm = min(warm_tokens, prompt_tokens)
        cold = prompt_tokens - warm
        prefill = cold / self.prefill_tok_s + self.attention_s_per_token2 * (
            prompt_tokens * prompt_tokens - warm * warm)
        return prefill + output_tokens / self.decode_tok_s


def adapt_anthropic(body):
    """Anthropic keeps the system turn in a top level field."""
    return {"system": body.get("system"),
            "messages": body.get("messages") or [],
            "max_tokens": body.get("max_tokens")}


def adapt_openai_chat(body):
    """OpenAI keeps the system turn inside messages, so it needs no hoisting.

    Prefix reuse below the frontend is identical either way: both endpoints parse
    into the same ChatTurn list, and a five turn conversation measured the same
    cached token counts and latencies through each. So this is only a body shape
    difference, not a behavioural one.
    """
    return {"system": None,
            "messages": body.get("messages") or [],
            "max_tokens": body.get("max_completion_tokens") or body.get("max_tokens")}


# Paths that generate, and therefore need affinity, queueing and streaming.
# Everything else is proxied. /v1/responses is deliberately absent: its store is
# process local, so previous_response_id and the /v1/responses/{id} subpaths only
# resolve on the server that created the response, and spreading them across two
# cards would return 404s.
GENERATING_ENDPOINTS = {
    "/v1/messages": adapt_anthropic,
    "/v1/chat/completions": adapt_openai_chat,
}


def prefix_hashes(system, messages):
    """Cumulative hashes of every conversation prefix, shortest first.

    The server caches exact prefixes, so turn N+1 of a conversation shares turn
    N's bytes. Hashing each cumulative prefix lets the router match a follow-up
    to the card that served the turn before it.
    """
    digest = hashlib.blake2b(digest_size=16)
    if system is not None:
        digest.update(json.dumps(system, sort_keys=True).encode())
    out = []
    chars = len(json.dumps(system)) if system is not None else 0
    for message in messages:
        digest.update(json.dumps(message, sort_keys=True).encode())
        chars += len(json.dumps(message))
        out.append((digest.hexdigest(), int(chars / CHARS_PER_TOKEN)))
    return out


def estimate_prompt_tokens(request):
    system = request["system"]
    chars = len(json.dumps(system)) if system is not None else 0
    chars += sum(len(json.dumps(m)) for m in request["messages"])
    return int(chars / CHARS_PER_TOKEN)


class Router:
    def __init__(self, backends, log):
        self.backends = backends
        self.log = log
        self.lock = threading.Lock()

    def choose(self, request):
        prompt_tokens = estimate_prompt_tokens(request)
        output_tokens = request["max_tokens"] or 512
        hashes = prefix_hashes(request["system"], request["messages"])

        with self.lock:
            fits = [b for b in self.backends
                    if prompt_tokens + output_tokens <= b.max_context]
            if not fits:
                # Nothing fits the estimate. Send it to the largest context and
                # let the server return its own error rather than inventing one.
                fits = [max(self.backends, key=lambda b: b.max_context)]
                reason = "no-fit"
            else:
                reason = None

            warm = {b.name: b.warm_tokens(hashes) for b in fits}
            best_warm = max(warm.values())

            if best_warm > 0:
                # Affinity wins outright unless the warm card is backed up badly
                # enough that re-prefilling elsewhere finishes sooner.
                warm_backends = [b for b in fits if warm[b.name] == best_warm]
                candidate = min(warm_backends, key=lambda b: b.queued)
                # Affinity is worth a wait, but not an unbounded one: compare
                # what each card would actually finish in, cache included.
                warm_cost = self._completion_estimate(
                    candidate, prompt_tokens, best_warm, output_tokens)
                others = [b for b in fits if b is not candidate]
                alternative = min(
                    others,
                    key=lambda b: self._completion_estimate(
                        b, prompt_tokens, warm[b.name], output_tokens),
                    default=None)
                if alternative is not None and self._completion_estimate(
                        alternative, prompt_tokens, warm[alternative.name],
                        output_tokens) < warm_cost:
                    chosen = alternative
                    reason = reason or "affinity-overridden-by-queue"
                else:
                    chosen = candidate
                    reason = reason or f"affinity({best_warm} tok)"
            elif any(prompt_tokens >= b.affinity_min_tokens for b in fits):
                # A cold prompt long enough to be retained will pull its own
                # follow up turns after it, so this is really a decision about
                # where a conversation lives. Each card holds only about three
                # prefixes, so spreading conversations matters more than which
                # card is momentarily quicker: piling a second conversation onto
                # an already warm card serialises both of them from then on.
                chosen = min(fits, key=lambda b: (
                    len(b.prefixes),
                    self._completion_estimate(b, prompt_tokens, warm[b.name], output_tokens)))
                reason = reason or "cold-sticky-spread"
            else:
                # Short and forgettable, so the only question is which card frees
                # up and finishes first. Scoring queue wait and service together
                # is what keeps a burst of short calls from all piling onto the
                # card with the faster decode.
                chosen = min(fits, key=lambda b: self._completion_estimate(
                    b, prompt_tokens, warm[b.name], output_tokens))
                reason = reason or ("cold-idle" if chosen.queued == 0
                                    else "cold-shortest-queue")

            committed = chosen.estimate_seconds(
                prompt_tokens, warm.get(chosen.name, 0), output_tokens)
            chosen.queued += 1
            chosen.pending_seconds += committed
            chosen.stats[reason.split("(")[0]] += 1
            chosen.remember(hashes)
        return chosen, reason, prompt_tokens, committed

    def _completion_estimate(self, backend, prompt_tokens, warm_tokens, output_tokens):
        """Work already committed to this backend, plus this request's own."""
        return backend.pending_seconds + backend.estimate_seconds(
            prompt_tokens, warm_tokens, output_tokens)

    def release(self, backend, committed):
        with self.lock:
            backend.queued -= 1
            backend.pending_seconds = max(0.0, backend.pending_seconds - committed)

    def snapshot(self):
        with self.lock:
            return {b.name: {"url": b.url, "queued": b.queued,
                             "pending_seconds": round(b.pending_seconds, 3),
                             "max_context": b.max_context,
                             "warm_prefixes": len(b.prefixes),
                             "routed": dict(b.stats)} for b in self.backends}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    router = None

    def log_message(self, fmt, *args):
        pass

    def _send_json(self, status, payload):
        blob = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def do_GET(self):
        if self.path == "/health":
            return self._send_json(200, {"status": "ok"})
        if self.path == "/router/status":
            return self._send_json(200, self.router.snapshot())
        return self._proxy_simple("GET", None)

    def do_POST(self):
        length = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(length) if length else b""
        adapter = GENERATING_ENDPOINTS.get(self.path)
        if adapter is not None:
            return self._proxy_generate(self.path, adapter, raw)
        return self._proxy_simple("POST", raw)

    def _pick_idle(self):
        with self.router.lock:
            return min(self.router.backends, key=lambda b: b.queued)

    def _proxy_simple(self, method, raw):
        """Anything that does not generate goes to the least loaded backend."""
        backend = self._pick_idle()
        req = urllib.request.Request(backend.url + self.path, data=raw, method=method,
                                     headers=self._forward_headers())
        try:
            # No short timeout here. This path still carries /v1/responses, which
            # generates, so a cap tuned for a metadata call would abort it.
            with urllib.request.urlopen(req, timeout=1800) as upstream:
                self._relay(upstream)
        except urllib.error.HTTPError as exc:
            self._relay_error(exc)
        except Exception as exc:
            self._send_json(502, {"error": {"message": f"upstream {backend.name}: {exc}"}})

    def _relay_error(self, exc):
        blob = exc.read()
        self.send_response(exc.code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def _relay(self, upstream):
        """Forward a response, streaming it if the upstream is streaming.

        Reading an event stream to completion before writing anything turns a
        streamed response into a single delivery at the end. Measured on an 80
        token completion, that moved the first chunk from 0.004s to 1.272s, which
        is the whole generation, so the check on content type is not cosmetic.
        """
        content_type = upstream.headers.get("content-type", "application/json")
        if "event-stream" not in content_type:
            blob = upstream.read()
            self.send_response(upstream.status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)
            return
        self.send_response(upstream.status)
        self.send_header("content-type", content_type)
        self.send_header("cache-control", "no-cache")
        self.send_header("transfer-encoding", "chunked")
        self.end_headers()
        for chunk in upstream:
            self.wfile.write(b"%x\r\n" % len(chunk) + chunk + b"\r\n")
            self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _forward_headers(self):
        keep = {}
        for name in ("content-type", "accept", "anthropic-version", "authorization"):
            value = self.headers.get(name)
            if value:
                keep[name] = value
        keep.setdefault("content-type", "application/json")
        return keep

    def _proxy_generate(self, path, adapter, raw):
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return self._send_json(400, {"error": {"message": "invalid json"}})
        if not isinstance(body, dict):
            return self._send_json(400, {"error": {"message": "body must be an object"}})

        backend, reason, prompt_tokens, committed = self.router.choose(adapter(body))
        started = time.perf_counter()
        try:
            # One in flight per backend. Holding the queue here rather than in
            # the server is what keeps a short request from waiting behind a
            # long prefill it never needed to follow.
            with backend.lock:
                waited = time.perf_counter() - started
                self._stream_from(backend, path, raw)
        finally:
            self.router.release(backend, committed)
        self.router.log(f"{backend.name:10s} {path:22s} {reason:32s} "
                        f"~{prompt_tokens:>7} tok wait={waited * 1000:7.1f}ms")

    def _stream_from(self, backend, path, raw):
        req = urllib.request.Request(backend.url + path, data=raw,
                                     headers=self._forward_headers())
        try:
            upstream = urllib.request.urlopen(req, timeout=1800)
        except urllib.error.HTTPError as exc:
            return self._relay_error(exc)
        except Exception as exc:
            return self._send_json(502, {"error": {"message": f"upstream {backend.name}: {exc}"}})

        with upstream:
            self._relay(upstream)


class ThreadedServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True


def parse_backend(spec):
    name, _, rest = spec.partition("=")
    parts = rest.split(",")
    url = parts[0]
    opts = dict(p.split("=", 1) for p in parts[1:] if "=" in p)
    return Backend(name, url,
                   int(opts.get("max_context", 8192)),
                   float(opts.get("prefill", 2000)),
                   float(opts.get("decode", 40)),
                   int(opts.get("affinity_min_tokens", AFFINITY_MIN_TOKENS)),
                   float(opts.get("attn", 0.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--backend", action="append", required=True,
                    help="name=url[,max_context=N][,prefill=tok/s][,decode=tok/s]"
                         "[,attn=s_per_token_squared][,affinity_min_tokens=N]")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    backends = [parse_backend(spec) for spec in args.backend]

    def log(line):
        if not args.quiet:
            print(line, file=sys.stderr, flush=True)

    Handler.router = Router(backends, log)
    server = ThreadedServer((args.host, args.port), Handler)
    log(f"router on http://{args.host}:{args.port} -> " +
        ", ".join(f"{b.name}({b.max_context})" for b in backends))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

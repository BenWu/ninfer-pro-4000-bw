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
  - The Pro 4000 holds 147456 tokens of context (its MTP-only default, no
    vision) against the 4090's 262144 (vision on), so long prompts are a
    placement constraint before they are a preference.

Shape only breaks ties between backends that are equally warm and equally idle.

Usage:
    python3 tools/router/ninfer_router.py --port 8090 \
        --backend pro4000=http://127.0.0.1:8080,max_context=147456,prefill=3860,decode=57.4,attn=2.493e-9,concurrency=1,slots_per_lane=3,vision=0 \
        --backend rtx4090=http://127.0.0.1:8081,max_context=262144,prefill=2115,decode=108.0,attn=1.619e-9,concurrency=2,slots_per_lane=1,vision=1
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

# An image is a few hundred to a couple of thousand tokens, but its base64 text
# is megabytes, so counting its characters would estimate a small screenshot at
# hundreds of thousands of tokens and conclude it fits nowhere. Both servers run
# --vision, so this case is live. Flat per-image estimate instead; use
# /v1/messages/count_tokens if an exact figure is ever needed.
TOKENS_PER_IMAGE = 1024

# What a server generates when the client names no limit. Matches the
# --default-max-tokens both launch scripts pass; guessing low here would
# underprice decode and bias placement toward the slower-decoding card.
DEFAULT_MAX_TOKENS = 8192

# How many contexts a card holds per lane. Both forks scale with the server's
# --max-concurrency, but by a different factor, so this is per backend.
#
# The Pro 4000 fork has a context cache whose capacities are derived from
# concurrency: two private continuations plus one shared prefix per lane
# (engine.cpp:62), so three. The 4090 fork has no such cache. Its reuse is per
# lane, against the sequence that lane retained, so it holds exactly one per
# lane: alternating two 4000 word documents paid the full 7.8s prefill every
# time at concurrency 1 and hit in 29ms at concurrency 2.
#
# Tracking more slots than a card has promises hits it cannot honour, so set
# concurrency to match the server and slots_per_lane to match its fork.
SLOTS_PER_LANE = 3

# How often to re-probe a backend that stopped answering.
HEALTH_PROBE_SECONDS = 5.0

# Retention is a cost model decision inside the engine, not a fixed rule, so this
# is a measured floor rather than a policy. Cycling two prompts on the Pro 4000
# with calibrated cost presets: 349 tokens never retained, 949 retained a third
# of the time, 1949 half, 3943 and above every time. Below this the router must
# not pin a request to a card for a hit the server will not give, because that
# only costs it balance. Re-measure after changing --max-concurrency or the
# context cost presets, both of which move the floor.
AFFINITY_MIN_TOKENS = 2048


class Backend:
    def __init__(self, name, url, max_context, prefill_tok_s, decode_tok_s,
                 affinity_min_tokens=AFFINITY_MIN_TOKENS, attention_s_per_token2=0.0,
                 affinity_slots=None, concurrency=1, slots_per_lane=SLOTS_PER_LANE,
                 vision=True):
        self.name = name
        self.url = url.rstrip("/")
        self.max_context = max_context
        self.prefill_tok_s = prefill_tok_s
        self.decode_tok_s = decode_tok_s
        # Prefill is not linear: measured on the Pro 4000, a flat tokens per
        # second rate underestimates a 63k prompt by 28%, and the gap widens
        # with length. Attention over the prompt is the quadratic part.
        self.attention_s_per_token2 = attention_s_per_token2
        self.lock = threading.Lock()      # one in flight, matching --max-concurrency 1
        self.queued = 0                   # waiting plus running, for placement
        self.prefixes = collections.OrderedDict()   # prefix hash -> token estimate
        self.stats = collections.Counter()
        self.affinity_min_tokens = affinity_min_tokens
        self.concurrency = max(1, concurrency)
        self.slots_per_lane = max(1, slots_per_lane)
        self.affinity_slots = max(1, affinity_slots if affinity_slots is not None
                                  else self.concurrency * self.slots_per_lane)
        # Set false when the backend stops answering, so a restart takes it out
        # of rotation instead of failing one request after another into it.
        self.healthy = True
        # Estimated work already committed to this backend. Counting jobs and
        # multiplying by an average badly misprices a queue whose jobs differ by
        # more than an order of magnitude, which is exactly the traffic here:
        # a cold 30k prefill and a chat turn are not interchangeable units.
        self.pending_seconds = 0.0
        # Vision capability: a backend running with vision disabled rejects
        # image content, so the router must not send it such a request.
        self.vision = vision

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
        while len(self.prefixes) > self.affinity_slots:
            self.prefixes.popitem(last=False)

    def forget(self, hashes):
        """Undo a remember() for a request that did not complete.

        Affinity is recorded at dispatch so a second request for the same
        conversation follows the first rather than racing it. If the request
        then fails, the card never built that prefix, and leaving the entry
        would send every follow up to a card holding nothing.
        """
        if not hashes:
            return
        self.prefixes.pop(hashes[-1][0], None)

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


def preamble_of(body):
    """Request fields that render ahead of the conversation.

    Tools become a "# Tools" block at the very front of the system preamble
    (`chat_template.cpp:321`), so two requests with identical messages but
    different tools share no prefix at all. Leaving them out of the hash would
    claim an affinity that misses from the first token.
    """
    keys = ("tools", "tool_choice", "response_format")
    present = {k: body[k] for k in keys if body.get(k) is not None}
    return present or None


def adapt_anthropic(body):
    """Anthropic keeps the system turn in a top level field."""
    return {"system": body.get("system"),
            "messages": body.get("messages") or [],
            "preamble": preamble_of(body),
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
            "preamble": preamble_of(body),
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


def has_vision(request):
    """True if any message in the request carries image content."""
    for message in request["messages"]:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") in ("image_url", "image", "input_image"):
                return True
    return False


def message_tokens(message):
    """Rough token size of one message, counting images flat rather than by text."""
    content = message.get("content")
    if not isinstance(content, list):
        return int(len(json.dumps(message)) / CHARS_PER_TOKEN)
    total = int(len(json.dumps(message.get("role", ""))) / CHARS_PER_TOKEN)
    for part in content:
        if not isinstance(part, dict):
            total += int(len(json.dumps(part)) / CHARS_PER_TOKEN)
        elif part.get("type") in ("image_url", "image", "input_image"):
            total += TOKENS_PER_IMAGE
        else:
            total += int(len(json.dumps(part)) / CHARS_PER_TOKEN)
    return total


def prefix_hashes(preamble, system, messages):
    """Cumulative hashes of every conversation prefix, shortest first.

    The server caches exact prefixes, so turn N+1 of a conversation shares turn
    N's bytes. Hashing each cumulative prefix lets the router match a follow-up
    to the card that served the turn before it. The preamble seeds the digest
    because it renders ahead of everything else.
    """
    digest = hashlib.blake2b(digest_size=16)
    if preamble is not None:
        digest.update(json.dumps(preamble, sort_keys=True).encode())
    if system is not None:
        digest.update(json.dumps(system, sort_keys=True).encode())
    out = []
    tokens = int(len(json.dumps(system)) / CHARS_PER_TOKEN) if system is not None else 0
    for message in messages:
        digest.update(json.dumps(message, sort_keys=True).encode())
        tokens += message_tokens(message)
        out.append((digest.hexdigest(), tokens))
    return out


def estimate_prompt_tokens(request):
    system = request["system"]
    total = int(len(json.dumps(system)) / CHARS_PER_TOKEN) if system is not None else 0
    if request["preamble"] is not None:
        total += int(len(json.dumps(request["preamble"])) / CHARS_PER_TOKEN)
    return total + sum(message_tokens(m) for m in request["messages"])


class Router:
    def __init__(self, backends, log):
        self.backends = backends
        self.log = log
        self.lock = threading.Lock()

    def choose(self, request, exclude=()):
        prompt_tokens = estimate_prompt_tokens(request)
        output_tokens = request["max_tokens"] or DEFAULT_MAX_TOKENS
        hashes = prefix_hashes(request["preamble"], request["system"], request["messages"])
        needs_vision = has_vision(request)

        with self.lock:
            usable = [b for b in self.backends if b not in exclude]
            if needs_vision:
                # A backend running without vision will reject image content,
                # so the only legal placements are the ones that can handle it.
                usable = [b for b in usable if b.vision]
            # Prefer backends that are answering, but if none are, try anyway
            # rather than refusing: a probe may simply not have caught up yet.
            answering = [b for b in usable if b.healthy]
            usable = answering or usable
            if not usable:
                return None, "no-backend", prompt_tokens, 0.0, hashes
            fits = [b for b in usable
                    if prompt_tokens + output_tokens <= b.max_context]
            no_fit = not fits
            if not fits:
                # Nothing fits the estimate. Send it to the largest context and
                # let the server return its own error rather than inventing one.
                fits = [max(usable, key=lambda b: b.max_context)]
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
            chosen.stats[("no-fit" if no_fit else reason.split("(")[0])] += 1
            chosen.remember(hashes)
        return chosen, reason, prompt_tokens, committed, hashes

    def _completion_estimate(self, backend, prompt_tokens, warm_tokens, output_tokens):
        """Work already committed to this backend, plus this request's own."""
        return backend.pending_seconds + backend.estimate_seconds(
            prompt_tokens, warm_tokens, output_tokens)

    def release(self, backend, committed, hashes=None, succeeded=True):
        with self.lock:
            backend.queued -= 1
            backend.pending_seconds = max(0.0, backend.pending_seconds - committed)
            if not succeeded:
                backend.forget(hashes)

    def mark(self, backend, healthy):
        with self.lock:
            if backend.healthy != healthy:
                backend.healthy = healthy
                self.log(f"{backend.name} is now {'up' if healthy else 'unreachable'}")

    def probe_unhealthy(self):
        """Restore backends that have started answering again."""
        for backend in self.backends:
            if backend.healthy:
                continue
            try:
                with urllib.request.urlopen(backend.url + "/health", timeout=2):
                    self.mark(backend, True)
            except Exception:
                pass

    def snapshot(self):
        with self.lock:
            return {b.name: {"url": b.url, "queued": b.queued,
                             "pending_seconds": round(b.pending_seconds, 3),
                             "max_context": b.max_context,
                             "warm_prefixes": f"{len(b.prefixes)}/{b.affinity_slots}",
                             "vision": b.vision,
                             "healthy": b.healthy,
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
            return True
        self.send_response(upstream.status)
        self.send_header("content-type", content_type)
        self.send_header("cache-control", "no-cache")
        self.send_header("transfer-encoding", "chunked")
        self.end_headers()
        try:
            for chunk in upstream:
                self.wfile.write(b"%x\r\n" % len(chunk) + chunk + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            # The client gave up. Returning here closes the upstream response,
            # which the server sees as its own client disconnecting and
            # cancels on, so the card is freed instead of generating into a
            # socket nobody is reading.
            self.close_connection = True
            return False
        return True

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

        request = adapter(body)
        tried = []
        last_error = None
        # Retry only while nothing has reached the client yet. Once the first
        # chunk is out, a second attempt would duplicate the response, so a
        # mid-stream failure has to stay a failure.
        for _ in range(len(self.router.backends)):
            backend, reason, prompt_tokens, committed, hashes = self.router.choose(
                request, exclude=tried)
            if backend is None:
                break
            started = time.perf_counter()
            waited = 0.0
            outcome = "failed"
            try:
                # One in flight per backend. Holding the queue here rather than
                # in the server is what keeps a short request from waiting
                # behind a long prefill it never needed to follow.
                with backend.lock:
                    waited = time.perf_counter() - started
                    outcome, last_error = self._attempt(backend, path, raw)
            finally:
                self.router.release(backend, committed, hashes, outcome == "ok")
            note = "" if outcome == "ok" else f"  [{outcome}]"
            self.router.log(f"{backend.name:10s} {path:22s} {reason:32s} "
                            f"~{prompt_tokens:>7} tok wait={waited * 1000:7.1f}ms{note}")
            if outcome != "retry":
                return
            tried.append(backend)

        if isinstance(last_error, urllib.error.HTTPError):
            return self._relay_error(last_error)
        self._send_json(502, {"error": {"message": f"no backend could serve the request: "
                                                   f"{last_error}"}})

    def _attempt(self, backend, path, raw):
        """One try against one backend.

        Returns (outcome, error). "retry" means nothing reached the client and
        another backend may serve it; "ok", "failed" and "client-gone" are all
        final.
        """
        req = urllib.request.Request(backend.url + path, data=raw,
                                     headers=self._forward_headers())
        try:
            upstream = urllib.request.urlopen(req, timeout=1800)
        except urllib.error.HTTPError as exc:
            # A 4xx is the request's own fault and would fail identically
            # elsewhere. A 5xx might not, so it is worth one more card.
            if exc.code >= 500:
                return "retry", exc
            self._relay_error(exc)
            return "failed", exc
        except Exception as exc:
            # Refused or dropped: the server is likely restarting.
            self.router.mark(backend, False)
            return "retry", exc

        self.router.mark(backend, True)
        with upstream:
            return ("ok", None) if self._relay(upstream) else ("client-gone", None)


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
                   float(opts.get("attn", 0.0)),
                   int(opts["affinity_slots"]) if "affinity_slots" in opts else None,
                   int(opts.get("concurrency", 1)),
                   int(opts.get("slots_per_lane", SLOTS_PER_LANE)),
                   vision=(opts.get("vision", "1") != "0"))


def backend_model_id(backend, timeout=5):
    """The model a backend reports, or None if it is not answering yet."""
    try:
        with urllib.request.urlopen(backend.url + "/v1/models", timeout=timeout) as response:
            payload = json.load(response)
    except Exception:
        return None
    entries = payload.get("data") or []
    return entries[0].get("id") if entries else None


def check_backends(backends, log):
    """Refuse to route across servers holding different models.

    Splitting one conversation between two models is not a slow answer, it is a
    wrong one, and nothing downstream would notice. A backend that is merely not
    up yet is only reported, since it may still be loading.
    """
    seen = {}
    for backend in backends:
        model = backend_model_id(backend)
        if model is None:
            log(f"warning: {backend.name} at {backend.url} is not answering /v1/models yet")
            continue
        seen[backend.name] = model
        log(f"{backend.name:10s} {backend.url}  model={model}")
    distinct = set(seen.values())
    if len(distinct) > 1:
        raise SystemExit(
            "backends serve different models, refusing to route: "
            + ", ".join(f"{n}={m}" for n, m in sorted(seen.items())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--backend", action="append", required=True,
                    help="name=url[,max_context=N][,prefill=tok/s][,decode=tok/s]"
                         "[,attn=s_per_token_squared][,affinity_min_tokens=N]"
                         "[,concurrency=N][,slots_per_lane=N][,affinity_slots=N]"
                         "[,vision=0|1] (default 1; set vision=0 on a backend "
                         "running without vision so image requests exclude it)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    backends = [parse_backend(spec) for spec in args.backend]

    def log(line):
        if not args.quiet:
            print(line, file=sys.stderr, flush=True)

    check_backends(backends, log)
    router = Router(backends, log)

    def probe_forever():
        while True:
            time.sleep(HEALTH_PROBE_SECONDS)
            router.probe_unhealthy()

    threading.Thread(target=probe_forever, daemon=True).start()
    Handler.router = router
    server = ThreadedServer((args.host, args.port), Handler)
    log(f"router on http://{args.host}:{args.port} -> " +
        ", ".join(f"{b.name}({b.max_context})" for b in backends))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

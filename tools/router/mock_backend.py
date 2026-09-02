#!/usr/bin/env python3
"""A stand-in ninfer server for exercising the router without two free GPUs.

It reproduces only the behaviours the router reacts to, all measured on this
machine: one request in flight at a time, an exact-prefix cache of a few
entries, and separate prefill and decode rates. It is not a model, and a green
run here is evidence about the router's choices, not about inference speed.
"""
import argparse
import collections
import hashlib
import json
import socketserver
import threading
import time
from http.server import BaseHTTPRequestHandler

CHARS_PER_TOKEN = 3.5
SLOTS = 3


class State:
    def __init__(self, prefill_tok_s, decode_tok_s, max_context, speed, model_id="mock-model"):
        self.prefill_tok_s = prefill_tok_s
        self.decode_tok_s = decode_tok_s
        self.max_context = max_context
        self.speed = speed          # wall clock divisor, so tests stay quick
        self.lock = threading.Lock()
        self.cache = collections.OrderedDict()
        self.served = 0
        self.hits = 0
        self.model_id = model_id


def prompt_tokens(body):
    chars = len(json.dumps(body.get("system"))) if body.get("system") is not None else 0
    chars += sum(len(json.dumps(m)) for m in body.get("messages", []))
    return int(chars / CHARS_PER_TOKEN)


def prefix_keys(body):
    """Every cumulative prefix, shortest first, with the tokens it covers.

    The engine reuses a matching prefix rather than only an identical request,
    so a follow up turn hits on everything before it. Matching exactly would
    hide the behaviour the router exists to exploit.
    """
    digest = hashlib.blake2b(digest_size=16)
    chars = 0
    if body.get("system") is not None:
        blob = json.dumps(body["system"], sort_keys=True)
        digest.update(blob.encode())
        chars += len(blob)
    out = []
    for message in body.get("messages", []):
        blob = json.dumps(message, sort_keys=True)
        digest.update(blob.encode())
        chars += len(blob)
        out.append((digest.hexdigest(), int(chars / CHARS_PER_TOKEN)))
    return out


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    state = None

    def log_message(self, fmt, *args):
        pass

    def handle_one_request(self):
        # Disconnect tests hang up mid-response on purpose; the default handler
        # prints a traceback for that, which is noise rather than a result.
        try:
            super().handle_one_request()
        except (ConnectionResetError, BrokenPipeError):
            self.close_connection = True

    def do_GET(self):
        if self.path == "/health":
            blob = json.dumps({"status": "ok"}).encode()
        elif self.path == "/v1/models":
            blob = json.dumps({"object": "list",
                               "data": [{"id": self.state.model_id, "object": "model"}]}).encode()
        else:
            blob = json.dumps({"served": self.state.served, "hits": self.state.hits}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def do_POST(self):
        length = int(self.headers.get("content-length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        state = self.state
        # A request the caller wants to fail, for exercising rollback paths.
        if "FAIL-ME" in json.dumps(body.get("messages") or []):
            blob = json.dumps({"error": {"message": "synthetic failure"}}).encode()
            self.send_response(500)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)
            return
        total = prompt_tokens(body)
        out = body.get("max_tokens") or 64
        keys = prefix_keys(body)

        with state.lock:                      # --max-concurrency 1
            cached = max((tokens for digest, tokens in keys if digest in state.cache),
                         default=0)
            cold = max(0, total - cached)
            if cached:
                state.hits += 1
            state.served += 1
            time.sleep(cold / state.prefill_tok_s / state.speed)
            ttft = time.perf_counter()
            time.sleep(out / state.decode_tok_s / state.speed)
            if keys:
                digest, tokens = keys[-1]
                state.cache.pop(digest, None)
                state.cache[digest] = tokens
            while len(state.cache) > SLOTS:
                state.cache.popitem(last=False)

        events = [
            {"type": "message_start", "message": {"usage": {
                "input_tokens": cold, "cache_read_input_tokens": cached}}},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "x"}},
            {"type": "message_delta", "usage": {"output_tokens": out}},
        ]
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("transfer-encoding", "chunked")
        self.end_headers()
        for event in events:
            chunk = f"data: {json.dumps(event)}\n\n".encode()
            self.wfile.write(b"%x\r\n" % len(chunk) + chunk + b"\r\n")
            self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()
        _ = ttft


class ThreadedServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True


def serve(port, prefill, decode, max_context, speed, model_id="mock-model"):
    handler = type("H", (Handler,),
                   {"state": State(prefill, decode, max_context, speed, model_id)})
    server = ThreadedServer(("127.0.0.1", port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, handler.state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--prefill", type=float, default=3350)
    ap.add_argument("--decode", type=float, default=57)
    ap.add_argument("--max-context", type=int, default=81920)
    ap.add_argument("--speed", type=float, default=50.0)
    args = ap.parse_args()
    serve(args.port, args.prefill, args.decode, args.max_context, args.speed)
    threading.Event().wait()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Measure greedy token agreement between two ninfer weight formats.

The question is how often the INT build drafts the token the NVFP4 build would
have produced. The mean committed tokens per round is the speedup ceiling for a
cross-GPU speculative pairing, before transport and scheduling overhead.

Two modes:

  record / replay   One server at a time, so the two models never need to be
                    resident together. This also lets both phases run the same
                    binary, leaving weights_id as the only variable.

  live              Both servers up at once, drafter and target queried per
                    round. Needs a drafter that supports assistant prefill.

Why record and replay is equivalent: in a greedy round every committed token is
the target's own greedy token. Accepted drafts match the target by definition,
and the correction is the target's token. So the agreed prefix is exactly the
target's greedy self-continuation and does not depend on the drafter. Recording
that trajectory once is enough to replay every round against it.

Servers must run with --no-thinking, and without --spec, since thinking blocks
and internal speculation both break the one delta per token assumption.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Iterator

DEFAULT_PROMPTS = [
    "Explain how a paged KV cache differs from a contiguous one.",
    "Write a Python function that merges two sorted lists.",
    "Summarize the tradeoffs between INT4 and FP4 weight quantization.",
    "Describe what happens during the prefill phase of LLM inference.",
    "List three reasons a GPU kernel might be memory bandwidth bound.",
]


class ProbeError(RuntimeError):
    pass


@dataclass
class Call:
    tokens: list[str]
    thinking_tokens: int
    output_tokens: int | None
    stop_reason: str | None
    seconds: float


@dataclass
class Round:
    accepted: int
    offered: int
    committed: int
    seconds: float
    position: int = 0


@dataclass
class PromptResult:
    prompt: str
    rounds: list[Round] = field(default_factory=list)
    note: str = ""


def post_sse(url: str, body: dict, timeout: float) -> Iterator[dict]:
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "content-type": "application/json",
            "accept": "text/event-stream",
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw in response:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:") :].strip()
                if payload:
                    yield json.loads(payload)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:400]
        raise ProbeError(f"{url} returned HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise ProbeError(f"cannot reach {url}: {error.reason}") from error


def generate(base_url: str, model: str, prompt: str, prefill: str, max_tokens: int,
             timeout: float) -> Call:
    """Generate greedily and return one entry per streamed token."""
    messages: list[dict] = [{"role": "user", "content": prompt}]
    if prefill:
        messages.append({"role": "assistant", "content": prefill})

    body = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "messages": messages,
    }

    tokens: list[str] = []
    thinking = 0
    output_tokens: int | None = None
    stop_reason: str | None = None

    started = time.perf_counter()
    for event in post_sse(f"{base_url.rstrip('/')}/v1/messages", body, timeout):
        kind = event.get("type")
        if kind == "content_block_delta":
            delta = event.get("delta") or {}
            if delta.get("type") == "text_delta":
                tokens.append(delta.get("text", ""))
            elif delta.get("type") == "thinking_delta":
                thinking += 1
        elif kind == "message_delta":
            usage = event.get("usage") or {}
            if "output_tokens" in usage:
                output_tokens = usage["output_tokens"]
            stop_reason = (event.get("delta") or {}).get("stop_reason", stop_reason)
        elif kind == "error":
            raise ProbeError(f"{base_url} streamed an error: "
                             f"{json.dumps(event.get('error', event))[:400]}")
    return Call(tokens=tokens, thinking_tokens=thinking, output_tokens=output_tokens,
                stop_reason=stop_reason, seconds=time.perf_counter() - started)


def resolve_model(base_url: str, override: str | None, timeout: float) -> str:
    if override:
        return override
    url = f"{base_url.rstrip('/')}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            listing = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError) as error:
        raise ProbeError(f"cannot list models at {url}: {error}") from error
    entries = listing.get("data") or []
    if not entries:
        raise ProbeError(f"{url} listed no models, pass an explicit model name")
    return entries[0]["id"]


def preflight(name: str, base_url: str, model: str, timeout: float) -> list[str]:
    """Check reachability and that one streamed delta means one token."""
    warnings: list[str] = []
    call = generate(base_url, model, "Count to five.", "The numbers are", 8, timeout)
    if call.thinking_tokens:
        warnings.append(f"{name} emitted thinking deltas, restart it with --no-thinking")
    if call.output_tokens is not None and call.tokens and call.output_tokens != len(call.tokens):
        warnings.append(
            f"{name} reported {call.output_tokens} output tokens but streamed "
            f"{len(call.tokens)} deltas, so deltas are not one token each "
            f"(restart without --spec)")
    if not call.tokens:
        warnings.append(
            f"{name} produced no tokens during preflight, which usually means the "
            f"assistant prefill continuation fix is missing from this build")
    return warnings


def accepted_prefix(drafts: list[str], reference: list[str]) -> int:
    count = 0
    for drafted, expected in zip(drafts, reference):
        if drafted != expected:
            break
        count += 1
    return count


def count_tokens(base_url: str, model: str, prompt: str, prefill: str, timeout: float) -> int:
    messages: list[dict] = [{"role": "user", "content": prompt}]
    if prefill:
        messages.append({"role": "assistant", "content": prefill})
    data = json.dumps({"model": model, "messages": messages}).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/messages/count_tokens", data=data,
        headers={"content-type": "application/json", "anthropic-version": "2023-06-01"},
        method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())["input_tokens"]
    except urllib.error.URLError as error:
        raise ProbeError(f"count_tokens failed at {base_url}: {error}") from error


def stable_boundaries(base_url: str, model: str, prompt: str, reference: list[str],
                      seed_text: str, timeout: float) -> list[bool]:
    """Mark prefix lengths whose joined text retokenizes back to the same token count.

    Generated tokens are streamed as text, so rejoining them can merge across a token
    boundary. Whitespace runs are the usual case: '\\n' followed by '\\n' comes back as a
    single '\\n\\n'. Replaying from such a prefix feeds the model a different token
    sequence than the one recorded, and the resulting disagreement says nothing about
    quantization. Those boundaries are skipped rather than measured.
    """
    counts = [count_tokens(base_url, model, prompt, seed_text, timeout)]
    for index in range(1, len(reference) + 1):
        joined = seed_text + "".join(reference[:index])
        counts.append(count_tokens(base_url, model, prompt, joined, timeout))

    # Test each step, not the running total. One merge shifts every later count by the
    # same offset, so an absolute test would discard the whole tail after the first one.
    stable = [True] * (len(reference) + 1)
    for index in range(1, len(reference) + 1):
        stable[index] = counts[index] - counts[index - 1] == 1
    return stable


# --- record -------------------------------------------------------------------


def record_by_decode(args: argparse.Namespace, model: str,
                     prompt: str) -> tuple[list[str], str | None, float]:
    """One continuous generation. Fast, but every token after the first comes from
    incremental decode, while replay probes each boundary with a prefill. Those two
    paths use different kernels and disagree on near-ties, which shows up as false
    rejections that have nothing to do with quantization."""
    call = generate(args.server, model, prompt, args.seed_text, args.max_tokens, args.timeout)
    return call.tokens, call.stop_reason, call.seconds


def record_by_prefill(args: argparse.Namespace, model: str,
                      prompt: str) -> tuple[list[str], str | None, float]:
    """One token per request, re-prefilling the accumulated text each step.

    This matches how replay probes a boundary, so the reference token at each position
    is the same computation replay performs. It also makes the trajectory self
    consistent under text joining: the model saw exactly the joined string, so a
    whitespace merge is already baked into the reference rather than being a mismatch.
    """
    tokens: list[str] = []
    stop_reason: str | None = None
    started = time.perf_counter()
    while len(tokens) < args.max_tokens:
        call = generate(args.server, model, prompt, args.seed_text + "".join(tokens), 1,
                        args.timeout)
        if not call.tokens:
            break
        tokens.append(call.tokens[0])
        stop_reason = call.stop_reason
        if stop_reason in {"end_turn", "stop_sequence"}:
            break
    return tokens, stop_reason, time.perf_counter() - started


def do_record(args: argparse.Namespace) -> int:
    model = resolve_model(args.server, args.model, args.timeout)
    print(f"target {args.server}  {model}")
    for warning in preflight("target", args.server, model, args.timeout):
        print(f"warning: {warning}", file=sys.stderr)

    prompts = load_prompts(args.prompts)
    trajectories = []
    for index, prompt in enumerate(prompts, start=1):
        print(f"[{index}/{len(prompts)}] {prompt.splitlines()[0][:60]}", flush=True)
        if args.regime == "decode":
            tokens, stop_reason, seconds = record_by_decode(args, model, prompt)
        else:
            tokens, stop_reason, seconds = record_by_prefill(args, model, prompt)
        if not tokens:
            print("  no tokens produced, skipped", file=sys.stderr)
            continue
        print(f"  {len(tokens)} tokens in {seconds:.1f}s")
        trajectories.append({
            "prompt": prompt,
            "tokens": tokens,
            "stop_reason": stop_reason,
        })

    if not trajectories:
        print("error: nothing recorded", file=sys.stderr)
        return 1

    payload = {
        "model": model,
        "server": args.server,
        "seed_text": args.seed_text,
        "regime": args.regime,
        "trajectories": trajectories,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    total = sum(len(t["tokens"]) for t in trajectories)
    print(f"\nwrote {args.out}: {len(trajectories)} trajectories, {total} tokens")
    print("Now stop this server, start the drafter weights, and run replay.")
    return 0


# --- replay -------------------------------------------------------------------


def do_replay(args: argparse.Namespace) -> int:
    with open(args.trajectories, encoding="utf-8") as handle:
        recorded = json.load(handle)

    model = resolve_model(args.server, args.model, args.timeout)
    seed_text = recorded.get("seed_text", "")
    print(f"drafter  {args.server}  {model}")
    print(f"recorded {recorded.get('model')} from {recorded.get('server')}")
    if model == recorded.get("model"):
        print("note: same model id on both sides. If these are the same weights the "
              "acceptance rate should come out near 100%, which is a good harness "
              "check but not a real measurement.")

    for warning in preflight("drafter", args.server, model, args.timeout):
        print(f"warning: {warning}", file=sys.stderr)

    # Both regimes need boundary filtering. A decode recording can retokenize
    # differently when its tokens are joined. A prefill recording bakes the merge into
    # the reference instead: the model re-prefilled the joined text at every step, so
    # the recorded tokens are not a sequence any continuous generation can reproduce.
    # Either way a round must not span a position whose prefix fails to round trip.
    regime = recorded.get("regime", "decode")
    check_boundaries = args.boundary_check != "off"
    print(f"regime   {regime} recording, boundary check "
          f"{'on' if check_boundaries else 'off'}")
    if regime == "decode":
        print("note: a decode-regime recording has a prefill/decode numeric floor. "
              "Record with --regime prefill for a clean baseline.")

    results: list[PromptResult] = []
    for index, entry in enumerate(recorded["trajectories"], start=1):
        prompt = entry["prompt"]
        reference: list[str] = entry["tokens"]
        print(f"[{index}/{len(recorded['trajectories'])}] "
              f"{prompt.splitlines()[0][:60]}", flush=True)

        result = PromptResult(prompt=prompt)
        stable = ([True] * (len(reference) + 1) if not check_boundaries else
                  stable_boundaries(args.server, model, prompt, reference, seed_text,
                                    args.timeout))
        skipped = 0
        position = 0
        while position < len(reference):
            # Only measure from a boundary whose text retokenizes to the recorded tokens.
            if not stable[position]:
                position += 1
                skipped += 1
                continue
            # A round is generated continuously, so every position it covers must round
            # trip. Stop the round before the next one that does not.
            limit = position + 1
            while (limit < len(reference) and limit - position < args.k
                   and stable[limit]):
                limit += 1
            budget = limit - position
            prefill = seed_text + "".join(reference[:position])
            try:
                call = generate(args.server, model, prompt, prefill, budget, args.timeout)
            except ProbeError as error:
                result.note = str(error)
                break
            if not call.tokens:
                result.note = "drafter produced no tokens"
                break

            accepted = accepted_prefix(call.tokens, reference[position:position + budget])
            # The round also commits the target's correction token, when the recorded
            # trajectory still has one at that position.
            committed = accepted + (1 if position + accepted < len(reference) else 0)
            result.rounds.append(Round(accepted=accepted, offered=len(call.tokens),
                                       committed=committed, seconds=call.seconds,
                                       position=position))
            if committed == 0:
                break
            position += committed

        results.append(result)
        if result.rounds:
            mean = statistics.fmean([r.accepted for r in result.rounds])
            note = f"  {result.note}" if result.note else ""
            unstable = f", {skipped} unstable boundaries skipped" if skipped else ""
            print(f"  {len(result.rounds)} rounds, mean accepted {mean:.2f}/{args.k}"
                  f"{unstable}{note}")

    return finish(results, args)


# --- live ---------------------------------------------------------------------


def do_live(args: argparse.Namespace) -> int:
    target_model = resolve_model(args.target, args.target_model, args.timeout)
    drafter_model = resolve_model(args.drafter, args.drafter_model, args.timeout)
    print(f"target   {args.target}  {target_model}")
    print(f"drafter  {args.drafter}  {drafter_model}")

    warnings = preflight("target", args.target, target_model, args.timeout)
    warnings += preflight("drafter", args.drafter, drafter_model, args.timeout)
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)

    results: list[PromptResult] = []
    for index, prompt in enumerate(load_prompts(args.prompts), start=1):
        print(f"[{index}] {prompt.splitlines()[0][:60]}", flush=True)
        result = PromptResult(prompt=prompt)
        agreed = args.seed_text
        committed_total = 0

        while committed_total < args.max_tokens:
            budget = min(args.k, args.max_tokens - committed_total)
            if budget < 1:
                break
            draft = generate(args.drafter, drafter_model, prompt, agreed, budget, args.timeout)
            if not draft.tokens:
                result.note = "drafter stopped producing tokens"
                break
            target = generate(args.target, target_model, prompt, agreed,
                              len(draft.tokens) + 1, args.timeout)
            if not target.tokens:
                result.note = "target stopped producing tokens"
                break

            accepted = accepted_prefix(draft.tokens, target.tokens)
            commit = (target.tokens[:accepted + 1]
                      if accepted < len(target.tokens) else target.tokens)
            agreed += "".join(commit)
            committed_total += len(commit)
            result.rounds.append(Round(accepted=accepted, offered=len(draft.tokens),
                                       committed=len(commit),
                                       seconds=draft.seconds + target.seconds))

            if len(target.tokens) < len(draft.tokens) + 1 or target.stop_reason in {
                    "end_turn", "stop_sequence"}:
                result.note = f"target finished ({target.stop_reason or 'end of stream'})"
                break

        results.append(result)
        if result.rounds:
            mean = statistics.fmean([r.accepted for r in result.rounds])
            print(f"  {len(result.rounds)} rounds, mean accepted {mean:.2f}/{args.k}")

    return finish(results, args)


# --- reporting ----------------------------------------------------------------


def positional_acceptance(rounds: list[Round], k: int) -> list[tuple[int, int, int]]:
    """Return (position, tested, accepted) using only rounds that reached each position."""
    table: list[tuple[int, int, int]] = []
    for position in range(k):
        tested = sum(1 for r in rounds if r.offered > position and r.accepted >= position)
        hits = sum(1 for r in rounds if r.accepted > position)
        if tested:
            table.append((position + 1, tested, hits))
    return table


def finish(results: list[PromptResult], args: argparse.Namespace) -> int:
    rounds = [entry for result in results for entry in result.rounds]
    if not rounds:
        print("error: no rounds completed, nothing to report", file=sys.stderr)
        return 1

    accepted = [r.accepted for r in rounds]
    committed = [r.committed for r in rounds]
    mean_accepted = statistics.fmean(accepted)
    mean_committed = statistics.fmean(committed)

    print()
    print(f"rounds            {len(rounds)} over {len(results)} prompts, K={args.k}")
    print(f"accepted drafts   mean {mean_accepted:.2f} of {args.k}")
    print(f"committed tokens  mean {mean_committed:.2f} per round")
    print(f"speedup ceiling   {mean_committed:.2f}x versus one token per step")
    print()

    print("acceptance by draft position")
    for position, tested, hits in positional_acceptance(rounds, args.k):
        rate = hits / tested
        print(f"  {position:>2}  {rate:6.1%}  ({hits}/{tested})  {'#' * round(rate * 40)}")
    print()

    histogram: dict[int, int] = {}
    for value in accepted:
        histogram[value] = histogram.get(value, 0) + 1
    print("accepted drafts per round")
    for value in sorted(histogram):
        share = histogram[value] / len(accepted)
        print(f"  {value:>2}  {share:6.1%}  ({histogram[value]})  {'#' * round(share * 40)}")
    print()

    verdict = (
        "worth building" if mean_committed >= 2.5 else
        "marginal, measure transport cost before committing" if mean_committed >= 1.8 else
        "not worth it, the quantizations disagree too often")
    print(f"verdict           {verdict}")

    if args.json:
        summary = {
            "k": args.k,
            "rounds": len(rounds),
            "mean_accepted": mean_accepted,
            "mean_committed": mean_committed,
            "acceptance_by_position": [
                {"position": p, "tested": t, "accepted": h}
                for p, t, h in positional_acceptance(rounds, args.k)],
            "histogram": histogram,
            "per_prompt": [
                {"prompt": r.prompt, "rounds": len(r.rounds), "note": r.note,
                 "accepted": [entry.accepted for entry in r.rounds],
                 "positions": [entry.position for entry in r.rounds],
                 "offered": [entry.offered for entry in r.rounds]}
                for r in results],
        }
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"wrote {args.json}")
    return 0


def load_prompts(path: str | None) -> list[str]:
    if not path:
        return DEFAULT_PROMPTS
    with open(path, encoding="utf-8") as handle:
        content = handle.read()
    if path.endswith(".jsonl"):
        prompts = []
        for line in content.splitlines():
            if line.strip():
                entry = json.loads(line)
                prompts.append(entry["prompt"] if isinstance(entry, dict) else str(entry))
        return prompts
    return [block.strip() for block in content.split("\n\n") if block.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure greedy agreement between two ninfer weight formats.")
    sub = parser.add_subparsers(dest="mode", required=True)

    def common(target: argparse.ArgumentParser) -> None:
        target.add_argument("--timeout", type=float, default=300.0)
        target.add_argument("--model", default=None, help="override the model id")

    record = sub.add_parser("record", help="record the target's greedy trajectories")
    record.add_argument("--server", default="http://127.0.0.1:8080",
                        help="the NVFP4 verifier, the only server that needs to be up")
    record.add_argument("--out", default="trajectories.json")
    record.add_argument("--prompts", default=None)
    record.add_argument("--max-tokens", type=int, default=192)
    record.add_argument("--seed-text", default="Sure.",
                        help="assistant prefill for the first round, keeps thinking off")
    record.add_argument("--regime", choices=("prefill", "decode"), default="prefill",
                        help="prefill records one token per request so the reference "
                             "matches what replay computes; decode is one continuous "
                             "generation, faster but adds a numeric floor")
    common(record)

    replay = sub.add_parser("replay", help="replay recorded trajectories against the drafter")
    replay.add_argument("--server", default="http://127.0.0.1:8080",
                        help="the INT drafter, started after the target is stopped")
    replay.add_argument("--trajectories", default="trajectories.json")
    replay.add_argument("-k", "--k", type=int, default=8)
    replay.add_argument("--json", default=None)
    replay.add_argument("--boundary-check", choices=("auto", "on", "off"), default="auto",
                        help="skip boundaries whose text does not retokenize to the "
                             "recorded tokens; auto enables it only for decode-regime "
                             "recordings")
    common(replay)

    live = sub.add_parser("live", help="query both servers at once")
    live.add_argument("--target", default="http://127.0.0.1:8080")
    live.add_argument("--drafter", default="http://127.0.0.1:8081")
    live.add_argument("--target-model", default=None)
    live.add_argument("--drafter-model", default=None)
    live.add_argument("-k", "--k", type=int, default=8)
    live.add_argument("--max-tokens", type=int, default=192)
    live.add_argument("--prompts", default=None)
    live.add_argument("--seed-text", default="Sure.")
    live.add_argument("--json", default=None)
    live.add_argument("--timeout", type=float, default=300.0)
    live.set_defaults(model=None)

    args = parser.parse_args()
    if getattr(args, "k", 1) < 1:
        parser.error("--k must be at least 1")

    handlers = {"record": do_record, "replay": do_replay, "live": do_live}
    try:
        return handlers[args.mode](args)
    except ProbeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

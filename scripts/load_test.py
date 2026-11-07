#!/usr/bin/env python3
"""Load and soak test for a running API: latency, throughput, errors, memory, and recovery.

Run a fake OpenAI-compatible LLM, so the test measures this service rather than an LLM bill:

    python -m scripts.load_test fake-llm --port 9100 --latency 0.8

Start the API with ``OPENAI_BASE_URL=http://127.0.0.1:9100/v1`` (any ``OPENAI_API_KEY``), then:

    python -m scripts.load_test run --url http://127.0.0.1:8000 --concurrency 16 --duration 300 \\
        --pid <API process id>

Phases: a sequential baseline, the concurrent soak, then a sequential recovery check. It exits
non-zero when the soak breaks the thresholds, so it can gate a release.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

TOPICS = (
    "hybrid search",
    "the reranker",
    "retrieval augmented generation",
    "evaluation",
    "BM25 scoring",
    "vector similarity",
    "citations",
    "self-healing",
    "chunking",
    "embeddings",
)
TEMPLATES = (
    "What is {t}?",
    "How does {t} work in this system?",
    "Explain {t} in two sentences.",
    "What are the limits of {t}?",
    "Give an example of {t}.",
    "Why use {t}? Case {n}.",
)


def questions(count: int, seed: int = 7) -> list[str]:
    """Distinct questions. A soak that sends more requests than this repeats them, so caches
    see both misses and hits rather than one hot key."""
    rng = random.Random(seed)
    return [
        f"{rng.choice(TEMPLATES).format(t=rng.choice(TOPICS), n=i)} (ref {i})" for i in range(count)
    ]


def percentile(values: Sequence[float], pct: float) -> float | None:
    """Nearest-rank percentile; ``None`` for no values."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, min(len(ordered), math.ceil(pct / 100 * len(ordered))))
    return ordered[rank - 1]


@dataclass
class Phase:
    name: str
    started: float = 0.0
    ended: float = 0.0
    latencies_ms: list[float] = field(default_factory=list)
    outcomes: Counter[str] = field(default_factory=Counter)
    # (seconds since start, ok) for every request, to find when errors started and stopped.
    timeline: list[tuple[float, bool]] = field(default_factory=list)

    @property
    def requests(self) -> int:
        return sum(self.outcomes.values())

    @property
    def errors(self) -> int:
        return self.requests - self.outcomes["200"]

    def summary(self) -> dict[str, object]:
        seconds = max(self.ended - self.started, 1e-9)
        return {
            "requests": self.requests,
            "per_second": round(self.requests / seconds, 2),
            "error_rate": round(self.errors / self.requests, 4) if self.requests else 0.0,
            "outcomes": dict(self.outcomes),
            "p50_ms": percentile(self.latencies_ms, 50),
            "p95_ms": percentile(self.latencies_ms, 95),
            "p99_ms": percentile(self.latencies_ms, 99),
            "max_ms": max(self.latencies_ms, default=None),
            "error_windows": error_windows(self.timeline),
        }


def error_windows(
    timeline: Sequence[tuple[float, bool]], gap: float = 5.0
) -> list[dict[str, float]]:
    """Spans where requests failed, merged when less than ``gap`` seconds apart."""
    windows: list[list[float]] = []
    for at, ok in sorted(timeline):
        if ok:
            continue
        if windows and at - windows[-1][1] <= gap:
            windows[-1][1] = at
        else:
            windows.append([at, at])
    return [{"from_s": round(a, 1), "to_s": round(b, 1)} for a, b in windows]


def _windows_children(pid: int) -> list[int]:
    import ctypes
    from ctypes import wintypes

    class Entry(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", ctypes.c_char * 260),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    snapshot = kernel.CreateToolhelp32Snapshot(0x2, 0)  # TH32CS_SNAPPROCESS
    children: list[int] = []
    entry = Entry()
    entry.dwSize = ctypes.sizeof(Entry)
    try:
        ok = kernel.Process32First(snapshot, ctypes.byref(entry))
        while ok:
            if entry.th32ParentProcessID == pid:
                children.append(entry.th32ProcessID)
            ok = kernel.Process32Next(snapshot, ctypes.byref(entry))
    finally:
        kernel.CloseHandle(snapshot)
    return children


def rss_mb(pid: int) -> float | None:
    """Resident memory of a local process, on Linux or Windows.

    On Windows a virtualenv's ``python.exe`` is a small launcher that runs the real
    interpreter as its child, so a tiny process with children is followed to its largest child.
    """
    own = _own_rss_mb(pid)
    if sys.platform == "win32" and own is not None and own < 30:
        children = [m for m in (_own_rss_mb(c) for c in _windows_children(pid)) if m is not None]
        if children:
            return max(children)
    return own


def _own_rss_mb(pid: int) -> float | None:
    status = Path(f"/proc/{pid}/status")
    if status.exists():
        match = re.search(r"VmRSS:\s+(\d+) kB", status.read_text())
        return int(match[1]) / 1024 if match else None
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t)
                for name in (
                    "PeakWorkingSetSize",
                    "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage",
                    "PagefileUsage",
                    "PeakPagefileUsage",
                )
            ]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = wintypes.HANDLE
        handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return None
        try:
            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(Counters),
                wintypes.DWORD,
            ]
            if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                return counters.WorkingSetSize / 1024 / 1024
        finally:
            kernel.CloseHandle(handle)
    return None


async def metrics_rss_mb(client: httpx.AsyncClient) -> float | None:
    """The API's own resident memory from /metrics (Linux only)."""
    try:
        text = (await client.get("/metrics")).text
    except httpx.HTTPError:
        return None
    match = re.search(r"^process_resident_memory_bytes ([0-9.e+]+)$", text, re.M)
    return float(match[1]) / 1024 / 1024 if match else None


def memory_trend(samples: Sequence[tuple[float, float]]) -> float | None:
    """MB per minute over the second half of the soak, after start-up allocations settle."""
    half = [s for s in samples if s[0] >= samples[-1][0] / 2] if samples else []
    if len(half) < 3:
        return None
    xs, ys = [t for t, _ in half], [m for _, m in half]
    mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
    var = sum((x - mean_x) ** 2 for x in xs)
    if not var:
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / var
    return round(slope * 60, 2)


async def one_request(
    client: httpx.AsyncClient, body: dict, phase: Phase, origin: float, timeout: float
) -> None:
    start = time.perf_counter()
    try:
        response = await client.post("/query", json=body, timeout=timeout)
        outcome = str(response.status_code)
    except httpx.TimeoutException:
        outcome = "timeout"
    except httpx.HTTPError as exc:
        outcome = type(exc).__name__
    elapsed = (time.perf_counter() - start) * 1000
    phase.outcomes[outcome] += 1
    phase.timeline.append((time.perf_counter() - origin, outcome == "200"))
    if outcome == "200":
        phase.latencies_ms.append(elapsed)


def request_body(question: str, args: argparse.Namespace, i: int) -> dict:
    return {
        "question": question,
        "self_heal": False,
        "use_hybrid": args.hybrid_share > 0 and (i % 100) < args.hybrid_share * 100,
        "use_reranker": args.reranker,
    }


async def run(args: argparse.Namespace) -> int:
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    pool = questions(args.distinct_questions)
    limits = httpx.Limits(max_connections=args.concurrency + 4)
    async with httpx.AsyncClient(base_url=args.url, headers=headers, limits=limits) as client:
        if (await client.get("/healthz", timeout=10)).status_code != 200:
            print("The API isn't answering /healthz.", file=sys.stderr)
            return 2
        origin = time.perf_counter()
        phases = {name: Phase(name) for name in ("baseline", "soak", "recovery")}

        async def sequential(phase: Phase, count: int) -> None:
            phase.started = time.perf_counter()
            for i in range(count):
                await one_request(
                    client, request_body(pool[i % len(pool)], args, i), phase, origin, args.timeout
                )
            phase.ended = time.perf_counter()

        print(f"baseline: {args.baseline_requests} sequential requests", flush=True)
        await sequential(phases["baseline"], args.baseline_requests)

        soak = phases["soak"]
        memory: list[tuple[float, float]] = []
        stop_at = time.perf_counter() + args.duration
        counter = iter(range(10**9))

        async def worker() -> None:
            while time.perf_counter() < stop_at:
                i = next(counter)
                await one_request(
                    client, request_body(pool[i % len(pool)], args, i), soak, origin, args.timeout
                )

        async def sample_memory() -> None:
            while time.perf_counter() < stop_at:
                value = rss_mb(args.pid) if args.pid else await metrics_rss_mb(client)
                if value is not None:
                    memory.append((time.perf_counter() - soak.started, value))
                await asyncio.sleep(args.memory_interval)

        async def progress() -> None:
            while time.perf_counter() < stop_at:
                await asyncio.sleep(15)
                last = memory[-1][1] if memory else None
                print(
                    f"  soak {time.perf_counter() - soak.started:5.0f}s: {soak.requests} requests, "
                    f"{soak.errors} errors" + (f", {last:.0f} MB" if last else ""),
                    flush=True,
                )

        print(f"soak: {args.concurrency} concurrent for {args.duration}s", flush=True)
        soak.started = time.perf_counter()
        await asyncio.gather(
            *(worker() for _ in range(args.concurrency)), sample_memory(), progress()
        )
        soak.ended = time.perf_counter()

        print(f"recovery: {args.baseline_requests} sequential requests", flush=True)
        await sequential(phases["recovery"], args.baseline_requests)

    report: dict[str, object] = {name: phase.summary() for name, phase in phases.items()}
    if memory:
        report["memory_mb"] = {
            "start": round(memory[0][1], 1),
            "peak": round(max(m for _, m in memory), 1),
            "end": round(memory[-1][1], 1),
            "trend_mb_per_min": memory_trend(memory),
        }
    problems = verdict(report, args)
    report["verdict"] = "pass" if not problems else "fail"
    report["problems"] = problems
    print(json.dumps(report, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if not problems else 1


def verdict(report: dict[str, object], args: argparse.Namespace) -> list[str]:
    problems = []
    soak, baseline, recovery = (report[k] for k in ("soak", "baseline", "recovery"))
    assert isinstance(soak, dict) and isinstance(baseline, dict) and isinstance(recovery, dict)
    if soak["error_rate"] > args.max_error_rate:
        problems.append(
            f"soak error rate {soak['error_rate']:.2%} is over {args.max_error_rate:.2%}"
        )
    if args.max_p95_ms and soak["p95_ms"] and soak["p95_ms"] > args.max_p95_ms:
        problems.append(f"soak p95 {soak['p95_ms']:.0f} ms is over {args.max_p95_ms:.0f} ms")
    if recovery["error_rate"] > 0:
        problems.append("requests still failed after the load stopped")
    if baseline["p50_ms"] and recovery["p50_ms"] and recovery["p50_ms"] > 2 * baseline["p50_ms"]:
        problems.append(
            f"latency didn't recover: p50 {recovery['p50_ms']:.0f} ms after vs "
            f"{baseline['p50_ms']:.0f} ms before"
        )
    memory = report.get("memory_mb")
    if isinstance(memory, dict) and memory["trend_mb_per_min"] is not None:
        if memory["trend_mb_per_min"] > args.max_memory_growth:
            problems.append(
                f"memory still growing {memory['trend_mb_per_min']} MB/min in the second half "
                f"(limit {args.max_memory_growth})"
            )
    return problems


def fake_llm(args: argparse.Namespace) -> None:
    """A minimal OpenAI-compatible chat endpoint with a fixed delay, for load tests."""
    import uvicorn

    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        body = await request.json()
        await asyncio.sleep(args.latency)
        text = "This answer comes from the load-test LLM, citing the first source [1]."
        created = int(time.time())
        if body.get("stream"):

            async def events():
                for word in text.split(" "):
                    chunk = {
                        "id": "x",
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": body.get("model", "fake"),
                        "choices": [
                            {"index": 0, "delta": {"content": word + " "}, "finish_reason": None}
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(events(), media_type="text/event-stream")
        return {
            "id": "x",
            "object": "chat.completion",
            "created": created,
            "model": body.get("model", "fake"),
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": text},
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        }

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)

    llm = commands.add_parser("fake-llm", help="Serve a fake OpenAI-compatible LLM")
    llm.add_argument("--host", default="127.0.0.1")
    llm.add_argument("--port", type=int, default=9100)
    llm.add_argument("--latency", type=float, default=0.8, help="Seconds per completion")

    load = commands.add_parser("run", help="Run baseline, soak, and recovery phases")
    load.add_argument("--url", default=os.getenv("RAG_API_URL", "http://127.0.0.1:8000"))
    load.add_argument("--api-key", default=os.getenv("RAG_API_KEY", ""))
    load.add_argument("--concurrency", type=int, default=8)
    load.add_argument("--duration", type=float, default=60.0, help="Soak length in seconds")
    load.add_argument("--baseline-requests", type=int, default=10)
    load.add_argument("--distinct-questions", type=int, default=1000)
    load.add_argument(
        "--hybrid-share", type=float, default=0.5, help="Share of requests using hybrid search"
    )
    load.add_argument("--reranker", action="store_true")
    load.add_argument("--timeout", type=float, default=60.0)
    load.add_argument("--pid", type=int, help="API process ID, to sample its memory locally")
    load.add_argument("--memory-interval", type=float, default=5.0)
    load.add_argument("--max-error-rate", type=float, default=0.01)
    load.add_argument("--max-p95-ms", type=float, default=0.0, help="0 disables the latency gate")
    load.add_argument("--max-memory-growth", type=float, default=10.0, help="MB per minute")
    load.add_argument("--json", help="Also write the report to this file")

    args = parser.parse_args(argv)
    if args.command == "fake-llm":
        fake_llm(args)
        return 0
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())

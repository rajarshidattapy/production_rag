"""Operational failure drill: real API and Chroma processes, killed and restarted on purpose.

Exercises the HTTP service the way an operator would see it, and prints what actually
happened next to what we would want to happen. Expectations are written down before the
run, so a failing row is a finding, not a broken script. Exit code is always 0.

Everything runs in temp directories, on loopback, against a local fake LLM: the API runs
outside the repo so it cannot read a developer's .env, and it refuses to start if LLM calls
would go anywhere else. Embeddings are a deterministic stub, so this measures service
behaviour and overhead, NOT retrieval quality, real LLM latency, or real embedding/reranker
cost. It does not exercise Docker.

Usage:
    python scripts/operational_drill.py
    python scripts/operational_drill.py --out drill.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUESTION = "What is hybrid search and how does reciprocal rank fusion combine the results?"
ANSWER = "Hybrid search combines keyword and vector retrieval using rank fusion [1]."
LOAD_REQUESTS = 300
LOAD_CONCURRENCY = 20
FAKE_LLM_DELAY_S = 0.05
HANG_CLIENT_TIMEOUT_S = 30.0
QUERY_DEADLINE_S = 20  # longer than the retry budget on an outage, or 504 masks the 503


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def kill_tree(proc: subprocess.Popen) -> None:
    """Hard-kill a process and its children, like a crash or an OOM kill."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, check=False
        )
    else:
        os.killpg(proc.pid, 9)
    proc.wait(timeout=20)


def spawn(cmd: list[str], env: dict[str, str] | None, log: Path, cwd: Path) -> subprocess.Popen:
    kwargs = {} if os.name == "nt" else {"start_new_session": True}
    return subprocess.Popen(
        cmd, cwd=cwd, env=env, stdout=log.open("ab"), stderr=subprocess.STDOUT, **kwargs
    )


# ---------------------------------------------------------------------------
# Fake OpenAI-compatible LLM
# ---------------------------------------------------------------------------


class FakeLLM:
    def __init__(self) -> None:
        self.mode = "ok"
        self.hits = 0
        self.release = threading.Event()
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                pass

            def do_POST(self) -> None:
                self.rfile.read(int(self.headers.get("content-length", 0)))
                fake.hits += 1
                if fake.mode == "hang":
                    fake.release.wait(timeout=600)
                if fake.mode in ("429", "500"):
                    self._send(int(fake.mode), {"error": {"message": fake.mode}})
                    return
                time.sleep(FAKE_LLM_DELAY_S)
                self._send(
                    200,
                    {
                        "id": "fake",
                        "object": "chat.completion",
                        "created": 0,
                        "model": "gpt-4o-mini",
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": "stop",
                                "message": {"role": "assistant", "content": ANSWER},
                            }
                        ],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    },
                )

            def _send(self, status: int, body: dict) -> None:
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                try:
                    self.wfile.write(data)
                except OSError:
                    pass  # the API gave up on this request

        self.port = free_port()
        self.server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self.release.set()
        self.server.shutdown()
        self.server.server_close()


class FakeOTLP:
    """Counts OTLP metric exports and Langfuse ingestion calls instead of storing them."""

    def __init__(self) -> None:
        self.otlp = 0
        self.langfuse = 0
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                pass

            def do_POST(self) -> None:
                self.rfile.read(int(self.headers.get("content-length", 0)))
                if self.path.startswith("/v1/metrics"):
                    fake.otlp += 1
                elif self.path.startswith("/api/public"):
                    fake.langfuse += 1
                self.send_response(207 if self.path.startswith("/api/public") else 200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

        self.port = free_port()
        self.server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()


# ---------------------------------------------------------------------------
# API process (--serve-api), with the embedding model stubbed out
# ---------------------------------------------------------------------------


def assert_llm_stays_local() -> None:
    """Exit unless every LLM call would land on the local fake provider.

    A developer's settings, such as RAG_OPENROUTER_MODE in .env, must never be able to send
    drill traffic to a real provider. An earlier version of this drill did exactly that.
    """
    sys.path.insert(0, str(ROOT))
    from src.generation.llm_client import LLMClient

    base_url = LLMClient("openai", "gpt-4o-mini").openai_client_kwargs().get("base_url")
    env_url = os.environ.get("OPENAI_BASE_URL", "")
    if base_url or not env_url.startswith("http://127.0.0.1:"):
        raise SystemExit(f"Refusing to start: LLM calls would go to {base_url or env_url!r}.")


def serve_api(port: int) -> None:
    assert_llm_stays_local()
    sys.path.insert(0, str(ROOT))
    import uvicorn

    from src.retrieval import vector_store

    def encode(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
            rng = random.Random(seed)
            out.append([rng.gauss(0.0, 1.0) for _ in range(384)])
        return out

    vector_store._ChromaEmbeddingFunction._encode = encode
    uvicorn.run("src.api:app", host="127.0.0.1", port=port, log_level="warning")


class Harness:
    def __init__(self, tmp: Path, llm: FakeLLM) -> None:
        self.tmp = tmp
        self.llm = llm
        self.chroma_port = free_port()
        self.chroma_dir = tmp / "chroma_server"
        self.chroma: subprocess.Popen | None = None
        self.results: list[dict] = []

    def check(self, scenario: str, name: str, expected: str, observed: str, ok: bool) -> None:
        self.results.append(
            {
                "scenario": scenario,
                "check": name,
                "expected": expected,
                "observed": observed,
                "ok": ok,
            }
        )
        print(f"[{'ok' if ok else 'FINDING'}] {scenario} / {name}: {observed}", flush=True)

    def start_chroma(self) -> None:
        exe = shutil.which("chroma")
        if exe is None:
            raise SystemExit("The `chroma` CLI is required (pip install chromadb).")
        self.chroma = spawn(
            [
                exe,
                "run",
                "--host",
                "127.0.0.1",
                "--path",
                str(self.chroma_dir),
                "--port",
                str(self.chroma_port),
            ],
            {**os.environ, "ANONYMIZED_TELEMETRY": "False"},
            self.tmp / "chroma.log",
            self.tmp,
        )

    def api_env(self, data_dir: Path) -> dict[str, str]:
        env = dict(os.environ)
        env.update(
            RAG_OPENROUTER_MODE="false",
            RAG_QUERY_TIMEOUT_SECONDS=str(QUERY_DEADLINE_S),
            ANONYMIZED_TELEMETRY="False",
            NO_PROXY="127.0.0.1",
            RAG_DATA_DIR=str(data_dir),
            RAG_CHROMA_PATH=str(data_dir / "chroma_db"),
            RAG_CHROMA_HOST="127.0.0.1",
            RAG_CHROMA_PORT=str(self.chroma_port),
            RAG_LLM_PROVIDER="openai",
            RAG_LLM_MODEL="gpt-4o-mini",
            RAG_API_KEY="",
            RAG_SELF_HEALING_ENABLED="false",
            MONITOR_ENABLED="false",
            OPENAI_API_KEY="drill-not-a-real-key",
            OPENAI_BASE_URL=f"http://127.0.0.1:{self.llm.port}/v1",
            PYTHONPATH=str(ROOT),
        )
        return env

    def start_api(
        self, port: int, data_dir: Path, extra_env: dict[str, str] | None = None
    ) -> subprocess.Popen:
        return spawn(
            [sys.executable, str(Path(__file__).resolve()), "--serve-api", str(port)],
            {**self.api_env(data_dir), **(extra_env or {})},
            self.tmp / f"api_{port}.log",
            self.tmp,  # not the repo root, so a developer's .env is never read
        )


def wait_healthy(client, base: str, timeout: float = 180.0) -> float:
    start = time.perf_counter()
    while time.perf_counter() - start < timeout:
        try:
            if client.get(f"{base}/healthz", timeout=2).status_code == 200:
                return time.perf_counter() - start
        except Exception:
            time.sleep(0.25)
    raise TimeoutError(f"{base} did not become healthy in {timeout}s")


def wait_chroma(client, port: int, timeout: float = 60.0) -> None:
    start = time.perf_counter()
    while time.perf_counter() - start < timeout:
        try:
            if client.get(f"http://127.0.0.1:{port}/api/v2/heartbeat", timeout=2).is_success:
                return
        except Exception:
            time.sleep(0.25)
    raise TimeoutError(f"Chroma did not start on port {port} within {timeout}s")


def timed(fn):
    start = time.perf_counter()
    try:
        resp = fn()
        return resp.status_code, time.perf_counter() - start, resp
    except Exception as exc:
        return type(exc).__name__, time.perf_counter() - start, None


def load_test(call) -> tuple[dict[str, int], float, float, float]:
    """Fire LOAD_REQUESTS calls at LOAD_CONCURRENCY; return status counts, req/s, p50, p95."""
    lat: list[float] = []
    codes: dict[str, int] = {}
    lock = threading.Lock()

    def one(_: int) -> None:
        status, secs, _resp = timed(call)
        with lock:
            lat.append(secs)
            codes[str(status)] = codes.get(str(status), 0) + 1

    start = time.perf_counter()
    with ThreadPoolExecutor(LOAD_CONCURRENCY) as pool:
        list(pool.map(one, range(LOAD_REQUESTS)))
    wall = time.perf_counter() - start
    p = statistics.quantiles(lat, n=100)
    return codes, LOAD_REQUESTS / wall, p[49], p[94]


def run(out: str | None) -> None:
    import httpx

    tmp = Path(tempfile.mkdtemp(prefix="rag_drill_"))
    llm = FakeLLM()
    h = Harness(tmp, llm)
    client = httpx.Client(timeout=120)
    data_a, data_b = tmp / "data_a", tmp / "data_b"
    data_b.mkdir(parents=True)
    shutil.copytree(ROOT / "data" / "sample_docs", data_a / "sample_docs")
    port_a, port_b = free_port(), free_port()
    a, b = f"http://127.0.0.1:{port_a}", f"http://127.0.0.1:{port_b}"
    api_a = api_b = None

    def ask(base: str, timeout: float = 120.0, **extra):
        body = {"question": QUESTION, "top_k": 3, **extra}
        return client.post(f"{base}/query", json=body, timeout=timeout)

    try:
        # ---- API starts while the database is down --------------------------------
        api_a = h.start_api(port_a, data_a)
        wait_healthy(client, a)
        status, secs, _ = timed(lambda: ask(a, timeout=60))
        h.check(
            "Chroma down at startup",
            "POST /query before Chroma has ever been reachable",
            "503: a server-side outage, not bad input",
            f"{status} in {secs:.1f}s",
            status == 503,
        )
        kill_tree(api_a)

        # ---- Startup and the happy path -----------------------------------------
        h.start_chroma()
        wait_chroma(client, h.chroma_port)
        api_a = h.start_api(port_a, data_a)
        startup = wait_healthy(client, a)
        h.check("Startup", "API becomes healthy", "healthy", f"healthy after {startup:.1f}s", True)

        status, secs, resp = timed(
            lambda: client.post(
                f"{a}/ingest", json={"source": str(data_a / "sample_docs")}, timeout=120
            )
        )
        chunks = resp.json().get("chunks_ingested") if status == 200 else 0
        h.check(
            "Startup",
            "POST /ingest (sample docs)",
            "200, chunks stored",
            f"{status}, {chunks} chunks in {secs:.2f}s",
            status == 200 and bool(chunks),
        )

        for label, extra in (("vector only", {}), ("hybrid", {"use_hybrid": True})):
            status, secs, resp = timed(lambda e=extra: ask(a, **e))
            cites = len(resp.json().get("citations", [])) if status == 200 else 0
            h.check(
                "Startup",
                f"POST /query ({label})",
                "200 with citations",
                f"{status}, {cites} citations in {secs:.2f}s",
                status == 200 and cites > 0,
            )

        status, secs, _ = timed(lambda: client.get(f"{a}/readyz", timeout=180))
        h.check(
            "Startup", "GET /readyz while healthy", "200", f"{status} in {secs:.1f}s", status == 200
        )

        # ---- Sustained concurrent requests ---------------------------------------
        for label, extra in (("vector only", {}), ("hybrid", {"use_hybrid": True})):
            codes, rps, p50, p95 = load_test(lambda e=extra: ask(a, **e))
            h.check(
                "Load",
                f"{LOAD_REQUESTS} /query at concurrency {LOAD_CONCURRENCY} ({label})",
                "all 200",
                f"{codes}; {rps:.0f} req/s, p50 {p50 * 1000:.0f} ms, p95 {p95 * 1000:.0f} ms "
                f"(stub embeddings, fake LLM {FAKE_LLM_DELAY_S * 1000:.0f} ms)",
                codes.get("200", 0) == LOAD_REQUESTS,
            )

        # ---- LLM provider failures ------------------------------------------------
        llm.mode = "429"
        status, secs, _ = timed(lambda: ask(a))
        h.check(
            "LLM failure",
            "Provider returns 429",
            "429 passed through",
            f"{status} in {secs:.1f}s",
            status == 429,
        )
        llm.mode = "500"
        status, secs, _ = timed(lambda: ask(a))
        h.check(
            "LLM failure",
            "Provider returns 500",
            "clean 5xx, no hang",
            f"{status} in {secs:.1f}s",
            isinstance(status, int) and 500 <= status < 600,
        )
        llm.mode = "hang"
        status, secs, _ = timed(lambda: ask(a, timeout=HANG_CLIENT_TIMEOUT_S))
        h.check(
            "LLM failure",
            f"Provider hangs (RAG_QUERY_TIMEOUT_SECONDS={QUERY_DEADLINE_S})",
            f"504 at the {QUERY_DEADLINE_S}s deadline",
            f"{status} after {secs:.1f}s",
            status == 504 and secs < QUERY_DEADLINE_S + 3,
        )
        llm.release.set()
        llm.mode = "ok"
        status, _, _ = timed(lambda: ask(a))
        h.check(
            "LLM failure",
            "Recovers once the provider is back",
            "200 without restart",
            str(status),
            status == 200,
        )

        # ---- Vector database outage ------------------------------------------------
        kill_tree(h.chroma)
        status, secs, _ = timed(lambda: client.get(f"{a}/healthz", timeout=10))
        h.check(
            "Chroma outage",
            "GET /healthz (liveness)",
            "200: the process is alive",
            f"{status} in {secs:.2f}s",
            status == 200,
        )
        status, secs, _ = timed(lambda: client.get(f"{a}/readyz", timeout=60))
        h.check(
            "Chroma outage",
            "GET /readyz (readiness)",
            "503: must not take traffic",
            f"{status} in {secs:.2f}s",
            status == 503,
        )
        status, secs, _ = timed(lambda: ask(a, timeout=90))
        h.check(
            "Chroma outage",
            "POST /query",
            "503 in a few seconds",
            f"{status} in {secs:.1f}s",
            status == 503 and secs < 30,
        )
        h.start_chroma()
        start = time.perf_counter()
        recovered = None
        while time.perf_counter() - start < 90:
            status, _, _ = timed(lambda: ask(a, timeout=30))
            if status == 200:
                recovered = time.perf_counter() - start
                break
            time.sleep(1)
        h.check(
            "Chroma outage",
            "Recovers after Chroma restarts",
            "200 without an API restart",
            f"200 after {recovered:.1f}s" if recovered is not None else "no 200 within 90s",
            recovered is not None,
        )

        # ---- API crash ---------------------------------------------------------------
        job = client.post(
            f"{a}/ingest/async", json={"source": str(data_a / "sample_docs")}, timeout=30
        ).json()["job_id"]
        for _ in range(120):
            state = client.get(f"{a}/ingest/jobs/{job}", timeout=10).json().get("status")
            if state in ("completed", "failed"):
                break
            time.sleep(0.5)
        before = client.get(f"{a}/stats", timeout=30).json().get("chunks_in_store")
        kill_tree(api_a)
        api_a = h.start_api(port_a, data_a)
        restart = wait_healthy(client, a)
        after = client.get(f"{a}/stats", timeout=30).json().get("chunks_in_store")
        h.check(
            "API crash",
            "Chunks survive kill -9 and restart",
            "same count",
            f"{before} before, {after} after; healthy in {restart:.1f}s",
            before == after,
        )
        status, secs, _ = timed(lambda: ask(a, use_hybrid=True))
        h.check(
            "API crash",
            "First hybrid /query after restart",
            "200",
            f"{status} in {secs:.2f}s",
            status == 200,
        )
        status, _, _ = timed(lambda: client.get(f"{a}/ingest/jobs/{job}", timeout=10))
        h.check(
            "API crash",
            "Async job status after restart",
            "job still known",
            f"{status} (job was '{state}' before the crash)",
            status == 200,
        )

        # ---- Two replicas on one Chroma, each with its own local disk -----------------
        api_b = h.start_api(port_b, data_b)
        wait_healthy(client, b)
        status, _, _ = timed(lambda: ask(b, use_hybrid=True))
        h.check("Replicas", "Replica B serves hybrid queries", "200", str(status), status == 200)
        token = "zebraquartzite"
        (data_a / "new.txt").write_text(
            f"The {token} protocol is documented only in this file.\n", "utf-8"
        )
        client.post(
            f"{a}/ingest", json={"source": str(data_a / "new.txt")}, timeout=60
        ).raise_for_status()
        resp = client.post(
            f"{b}/query",
            json={"question": f"What is the {token} protocol?", "use_hybrid": True},
            timeout=60,
        )
        seen = any(token in c["text_snippet"] for c in resp.json().get("citations", []))
        h.check(
            "Replicas",
            "Replica B finds a document ingested through A",
            "found",
            f"{resp.status_code}, {'found' if seen else 'not found'}",
            seen,
        )
        (data_a / "new2.txt").write_text(f"Replacement {token} text after a reset.\n", "utf-8")
        status, _, _ = timed(
            lambda: client.post(
                f"{a}/ingest", json={"source": str(data_a / "new2.txt"), "reset": True}, timeout=60
            )
        )
        status_b, _, _ = timed(lambda: ask(b))
        h.check(
            "Replicas",
            "Replica B after A resets the store (reset=true)",
            "200 without a restart",
            f"A reset: {status}; B /query: {status_b}",
            status_b == 200,
        )

        # ---- MONITOR_ENABLED=true actually reaches the HTTP service --------------------
        kill_tree(api_b)
        otlp = FakeOTLP()
        try:
            api_b = h.start_api(
                port_b,
                data_b,
                {
                    "MONITOR_ENABLED": "true",
                    "MONITOR_OTEL_EXPORTER_OTLP_ENDPOINT": f"http://127.0.0.1:{otlp.port}",
                    "MONITOR_OTEL_EXPORT_INTERVAL_MS": "1000",
                    "MONITOR_LANGFUSE_SECRET_KEY": "sk-lf-drill",
                    "MONITOR_LANGFUSE_PUBLIC_KEY": "pk-lf-drill",
                    "MONITOR_LANGFUSE_HOST": f"http://127.0.0.1:{otlp.port}",
                },
            )
            wait_healthy(client, b)
            status, _, _ = timed(lambda: ask(b))
            stream = client.post(f"{b}/query/stream", json={"question": QUESTION}, timeout=60)
            streamed = "citations" in stream.text
            time.sleep(8)  # both exporters flush on a timer
            h.check(
                "Monitoring",
                "MONITOR_ENABLED=true: /query and /query/stream work",
                "200, with citations in the stream",
                f"{status}; stream citations: {streamed}",
                status == 200 and streamed,
            )
            h.check(
                "Monitoring",
                "OTLP metrics reach the configured collector",
                "more than 0 exports",
                str(otlp.otlp),
                otlp.otlp > 0,
            )
            h.check(
                "Monitoring",
                "Langfuse traces reach the configured host",
                "more than 0 ingestion calls",
                str(otlp.langfuse),
                otlp.langfuse > 0,
            )
        finally:
            otlp.stop()
    finally:
        client.close()
        for proc in (api_a, api_b, h.chroma):
            if proc is not None:
                try:
                    kill_tree(proc)
                except Exception:
                    pass
        llm.stop()
        shutil.rmtree(tmp, ignore_errors=True)

    h.check(
        "Isolation", "LLM calls reached the local fake", "more than 0", str(llm.hits), llm.hits > 0
    )
    print("\n| Scenario | Check | Wanted | Observed | Met? |\n|---|---|---|---|---|")
    for r in h.results:
        met = "yes" if r["ok"] else "**no**"
        print(f"| {r['scenario']} | {r['check']} | {r['expected']} | {r['observed']} | {met} |")
    print(f"\n{sum(r['ok'] for r in h.results)}/{len(h.results)} checks met what we wanted.")
    if out:
        Path(out).write_text(json.dumps(h.results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", help="Also write the results to this JSON file")
    parser.add_argument("--serve-api", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.serve_api:
        serve_api(args.serve_api)
    else:
        run(args.out)

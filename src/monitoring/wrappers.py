from __future__ import annotations

import asyncio
import contextvars
import logging
import threading
import time
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

from src.utils.usage import UsageTracker, request_usage

from .extensions import (
    BaseExtension,
    LangfuseTracingExtension,
    OTelMetricsExtension,
)
from .tracing import Tracer

logger = logging.getLogger(__name__)


def _get_default_system_prompt() -> str:
    try:
        import importlib

        generator_mod = importlib.import_module("src.generation.generator")
        resolver = getattr(generator_mod, "get_default_system_prompt", None)
        if resolver is not None:
            return resolver()
        return getattr(generator_mod, "DEFAULT_SYSTEM_PROMPT", "")
    except ImportError:
        return ""


class _HookObserver:
    """Adapts the healing engine's observer protocol onto extension lifecycle hooks."""

    def __init__(self, run_hook: Any) -> None:
        self._run_hook = run_hook

    def on_step_start(
        self, name: str, input_data: dict[str, Any], metadata: dict[str, Any]
    ) -> None:
        self._run_hook("on_step_start", name, input_data, metadata)

    def on_step_end(
        self,
        name: str,
        output: dict[str, Any],
        elapsed: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._run_hook("on_step_end", name, output, elapsed, metadata=metadata)

    def on_step_error(self, name: str, exc: Exception) -> None:
        self._run_hook("on_step_error", name, exc)


def _count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens with tiktoken when available, else estimate from the length."""
    try:
        import importlib

        tiktoken = importlib.import_module("tiktoken")
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        return max(1, len(text) // 4)


class MonitoredRAGPipeline:
    """Instrumented wrapper around RAGPipeline.

    Adds Langfuse tracing and OTel metrics to every pipeline step
    via a pluggable extension architecture.

    Trace nesting and per-query token totals are thread-local, because the HTTP service
    runs concurrent queries, each on its own worker thread.
    """

    def __init__(
        self,
        pipeline: Any,
        tracer: Tracer | None = None,
        metrics: Any | None = None,
        extensions: list[BaseExtension] | None = None,
    ) -> None:
        self._usage_local = threading.local()
        self._pipeline = pipeline
        self.extensions = extensions or []

        # Backward compatibility for direct tracer/metrics configurations
        has_tracing = any(isinstance(e, LangfuseTracingExtension) for e in self.extensions)
        has_metrics = any(isinstance(e, OTelMetricsExtension) for e in self.extensions)

        if not has_tracing:
            self._tracer = tracer or Tracer()
            self.extensions.append(LangfuseTracingExtension(self._tracer))
        else:
            self._tracer = next(
                e.tracer for e in self.extensions if isinstance(e, LangfuseTracingExtension)
            )

        if not has_metrics:
            self._metrics = metrics
            self.extensions.append(OTelMetricsExtension(self._metrics))
        else:
            self._metrics = next(
                e.metrics for e in self.extensions if isinstance(e, OTelMetricsExtension)
            )

        # Initialize token/cost tracking at query level
        self._last_query_tokens = {"prompt": 0, "completion": 0, "total": 0}
        self._last_query_cost = 0.0

        self._instrument_pipeline()

    @property
    def _last_query_tokens(self) -> dict[str, int]:
        return getattr(self._usage_local, "tokens", {"prompt": 0, "completion": 0, "total": 0})

    @_last_query_tokens.setter
    def _last_query_tokens(self, value: dict[str, int]) -> None:
        self._usage_local.tokens = value

    @property
    def _last_query_cost(self) -> float:
        return getattr(self._usage_local, "cost", 0.0)

    @_last_query_cost.setter
    def _last_query_cost(self, value: float) -> None:
        self._usage_local.cost = value

    def __getattr__(self, name: str) -> Any:
        # The API reaches pipeline internals (stores, retrievers). Instrumented steps are
        # patched onto the wrapped pipeline itself, so delegation still goes through them.
        if name == "_pipeline":
            raise AttributeError(name)
        return getattr(self._pipeline, name)

    def _run_hook(self, ext_method_name: str, *args: Any, **kwargs: Any) -> None:
        for ext in self.extensions:
            hook = getattr(ext, ext_method_name, None)
            if hook is not None:
                try:
                    hook(*args, **kwargs)
                except ValueError:
                    raise
                except Exception as e:
                    logger.warning(
                        "Extension hook %s failed on %s: %s", ext_method_name, type(ext).__name__, e
                    )

    def _instrument_pipeline(self) -> None:
        if not hasattr(self._pipeline, "generator"):
            return

        # 1. Wrap _retrieve
        original_retrieve = self._pipeline._retrieve

        def monitored_retrieve(
            query: str,
            use_hybrid: bool = False,
            use_reranker: bool = False,
            k: int = 5,
            lang: str = "en",
            fetch_k: int | None = None,
            alpha: float | None = None,
            where: dict[str, Any] | None = None,
        ):
            start_time = time.monotonic()
            input_data = {
                "query": query,
                "use_hybrid": use_hybrid,
                "use_reranker": use_reranker,
                "k": k,
                "lang": lang,
            }
            # Self-healing retries pass these overrides; forward them only when
            # set so wrapped retrievers with the original signature still work.
            overrides = {
                name: value
                for name, value in (("fetch_k", fetch_k), ("alpha", alpha), ("where", where))
                if value is not None
            }
            input_data.update(overrides)

            self._run_hook("on_step_start", "retrieve", input_data, {})

            kwargs: dict[str, Any] = {
                "use_hybrid": use_hybrid,
                "use_reranker": use_reranker,
                "k": k,
                "lang": lang,
                **overrides,
            }

            try:
                contexts = original_retrieve(query, **kwargs)
            except Exception as exc:
                self._run_hook("on_step_error", "retrieve", exc)
                raise

            elapsed = time.monotonic() - start_time
            chunk_ids = [ctx.get("id") for ctx in contexts]
            retrieved_metadata = []
            for ctx in contexts:
                meta = ctx.get("metadata", {})
                source = meta.get("source", meta.get("filename", "unknown"))
                title = meta.get("title", meta.get("source", "unknown"))
                retrieved_metadata.append(
                    {
                        "id": ctx.get("id"),
                        "source": source,
                        "title": title,
                        "score": ctx.get("score"),
                    }
                )

            output_val = {"chunk_ids": chunk_ids, "chunks": retrieved_metadata}

            self._run_hook(
                "on_step_end",
                "retrieve",
                output_val,
                elapsed,
                metadata={"chunk_ids": chunk_ids, "chunks": retrieved_metadata},
            )

            return contexts

        self._pipeline._retrieve = monitored_retrieve

        # 2. Wrap _apply_reranker
        original_rerank = self._pipeline._apply_reranker

        def monitored_rerank(query: str, contexts: list[dict[str, Any]], top_k: int = 5):
            start_time = time.monotonic()
            input_data = {"query": query, "contexts_count": len(contexts), "top_k": top_k}

            self._run_hook("on_step_start", "rerank", input_data, {})

            try:
                reranked = original_rerank(query, contexts, top_k=top_k)
            except Exception as exc:
                self._run_hook("on_step_error", "rerank", exc)
                raise

            elapsed = time.monotonic() - start_time
            pre_scores = {ctx.get("id"): ctx.get("score") for ctx in contexts}
            post_scores = {ctx.get("id"): ctx.get("score") for ctx in reranked}
            selected_ids = [ctx.get("id") for ctx in reranked]

            model_name = (
                self._pipeline._reranker.model_name
                if self._pipeline._reranker is not None
                else "unknown"
            )

            output_metadata = {
                "model": model_name,
                "pre_scores": pre_scores,
                "post_scores": post_scores,
                "selected_ids": selected_ids,
            }

            self._run_hook(
                "on_step_end", "rerank", output_metadata, elapsed, metadata=output_metadata
            )

            return reranked

        self._pipeline._apply_reranker = monitored_rerank

        # 3. Wrap generator.generate
        original_generate = self._pipeline.generator.generate

        def monitored_generate(
            query: str, contexts: list[dict[str, Any]], system_prompt: str | None = None
        ) -> str:
            start_time = time.monotonic()

            DEFAULT_SYSTEM_PROMPT = _get_default_system_prompt()

            formatted_context = self._pipeline.generator._format_context(contexts)
            prompt_content = (system_prompt or DEFAULT_SYSTEM_PROMPT).format(
                context=formatted_context
            )

            input_data = {"query": query, "contexts_count": len(contexts)}
            self._run_hook("on_step_start", "generate", input_data, {})

            # Reuse the request-scoped tracker LLMClient already fills (src/utils/usage.py)
            # instead of separately patching the SDK clients. A fresh tracker, swapped in only
            # for this call, isolates its usage; its totals are folded back into any outer
            # tracker api.py set for the whole request, so the X-RAG-*-Tokens headers still see it.
            outer_tracker = request_usage.get()
            call_tracker = UsageTracker()
            usage_token = request_usage.set(call_tracker)
            try:
                answer = original_generate(query, contexts, system_prompt=system_prompt)
            except Exception as exc:
                request_usage.reset(usage_token)
                self._run_hook("on_step_error", "generate", exc)
                raise
            request_usage.reset(usage_token)
            if outer_tracker is not None:
                outer_tracker.add_call(
                    call_tracker.prompt_tokens,
                    call_tracker.completion_tokens,
                    call_tracker.total_latency,
                )
            usage_dict = {
                "prompt_tokens": call_tracker.prompt_tokens,
                "completion_tokens": call_tracker.completion_tokens,
                "total_tokens": call_tracker.total_tokens,
            }

            elapsed = time.monotonic() - start_time
            provider = self._pipeline.generator.provider
            model = self._pipeline.generator.model

            cost = self._tracer._pricing.get_cost(
                provider, model, usage_dict["prompt_tokens"], usage_dict["completion_tokens"]
            )

            self._last_query_tokens = {
                "prompt": usage_dict["prompt_tokens"],
                "completion": usage_dict["completion_tokens"],
                "total": usage_dict["total_tokens"],
            }
            self._last_query_cost = cost

            self._run_hook(
                "on_generation_llm_call",
                model=model,
                provider=provider,
                prompt=prompt_content,
                query=query,
                response=answer,
                usage=usage_dict,
                cost=cost,
            )
            self._run_hook("on_step_end", "generate", {"answer": answer}, elapsed)

            return answer

        self._pipeline.generator.generate = monitored_generate

        # 4. Route self-healing steps (retrieve_attempt_N, query_rewrite,
        #    verify_attempt_N, ...) through the same extension hooks so each
        #    attempt shows up as its own span under the query trace.
        self._pipeline.set_healing_observer(_HookObserver(self._run_hook))

    def ingest(self, source: Path | str, **options: Any) -> int:
        self._run_hook("on_step_start", "ingest", {"source": str(source)}, {})

        start_time = time.monotonic()
        try:
            res = self._pipeline.ingest(source, **options)
            elapsed = time.monotonic() - start_time
            self._run_hook("on_step_end", "ingest", {"chunks_ingested": res}, elapsed)
            return res
        except Exception as exc:
            self._run_hook("on_step_error", "ingest", exc)
            raise

    def query(
        self,
        question: str,
        top_k: int | None = None,
        use_hybrid: bool = False,
        use_reranker: bool = False,
        self_heal: bool | None = None,
    ) -> tuple[str, list[Any]]:
        start = time.monotonic()
        metadata = {
            "hybrid": use_hybrid,
            "reranker": use_reranker,
            "top_k": top_k,
        }
        # Only forward self_heal when set, so wrapped pipelines that predate
        # self-healing (or test doubles) keep receiving the original kwargs.
        extra: dict[str, Any] = {}
        if self_heal is not None:
            extra["self_heal"] = self_heal
            metadata["self_heal"] = self_heal

        self._last_query_tokens = {"prompt": 0, "completion": 0, "total": 0}
        self._last_query_cost = 0.0

        # Run query start guardrails and hooks
        self._run_hook("on_query_start", question, metadata)

        try:
            answer, citations = self._pipeline.query(
                question,
                top_k=top_k,
                use_hybrid=use_hybrid,
                use_reranker=use_reranker,
                **extra,
            )
        except Exception as exc:
            self._run_hook("on_query_error", exc)
            raise

        elapsed = time.monotonic() - start

        # Run query completion hooks
        self._run_hook(
            "on_query_end",
            answer,
            citations,
            elapsed,
            self._last_query_tokens,
            self._last_query_cost,
        )

        return answer, citations

    def query_with_healing(
        self,
        question: str,
        top_k: int | None = None,
        use_hybrid: bool = False,
        use_reranker: bool = False,
    ) -> Any:
        """Traced self-healing query.

        Produces a ``query`` root span whose children are the per-attempt
        healing steps (``retrieve_attempt_1``, ``verify_attempt_1``,
        ``query_rewrite``, ``retrieve_attempt_2``, ...).
        """
        start = time.monotonic()
        metadata = {
            "hybrid": use_hybrid,
            "reranker": use_reranker,
            "top_k": top_k,
            "self_heal": True,
        }
        self._last_query_tokens = {"prompt": 0, "completion": 0, "total": 0}
        self._last_query_cost = 0.0
        self._run_hook("on_query_start", question, metadata)
        try:
            result = self._pipeline.query_with_healing(
                question, top_k=top_k, use_hybrid=use_hybrid, use_reranker=use_reranker
            )
        except Exception as exc:
            self._run_hook("on_query_error", exc)
            raise
        self._run_hook(
            "on_query_end",
            result.answer,
            result.citations,
            time.monotonic() - start,
            self._last_query_tokens,
            self._last_query_cost,
        )
        return result

    async def query_async(
        self,
        question: str,
        top_k: int | None = None,
        use_hybrid: bool = False,
        use_reranker: bool = False,
        self_heal: bool | None = None,
    ) -> tuple[str, list[Any]]:
        """Traced query on a worker thread, so the whole trace nests on that one thread."""
        return await asyncio.to_thread(
            self.query,
            question,
            top_k=top_k,
            use_hybrid=use_hybrid,
            use_reranker=use_reranker,
            self_heal=self_heal,
        )

    async def query_with_healing_async(
        self,
        question: str,
        top_k: int | None = None,
        use_hybrid: bool = False,
        use_reranker: bool = False,
    ) -> Any:
        """Traced self-healing query on a worker thread."""
        return await asyncio.to_thread(
            self.query_with_healing,
            question,
            top_k=top_k,
            use_hybrid=use_hybrid,
            use_reranker=use_reranker,
        )

    async def stream_answer(
        self,
        question: str,
        top_k: int | None = None,
        use_hybrid: bool = False,
        use_reranker: bool = False,
    ) -> AsyncGenerator[tuple[str, Any], None]:
        """Traced ``RAGPipeline.stream_answer``.

        Spans are nested per thread, so every hook and blocking step of one stream runs on
        its own single-thread executor, while tokens arrive on the event loop. Token counts
        are estimates because streaming responses do not report usage.
        """
        loop = asyncio.get_running_loop()
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag-stream-trace")

        async def run_blocking(fn: Any, *args: Any, **kwargs: Any) -> Any:
            context = contextvars.copy_context()
            return await loop.run_in_executor(executor, partial(context.run, fn, *args, **kwargs))

        start = time.monotonic()
        generator = self._pipeline.generator
        provider = generator.provider
        model = generator.model
        metadata = {
            "hybrid": use_hybrid,
            "reranker": use_reranker,
            "top_k": top_k,
            "streaming": True,
        }
        contexts: list[Any] = []
        tokens: list[str] = []
        first_token_at: float | None = None
        generation_started = False

        try:
            await run_blocking(self._run_hook, "on_query_start", question, metadata)
            async for kind, value in self._pipeline.stream_answer(
                question,
                top_k=top_k,
                use_hybrid=use_hybrid,
                use_reranker=use_reranker,
                run_blocking=run_blocking,
            ):
                if kind == "contexts":
                    contexts = value
                    await run_blocking(
                        self._run_hook,
                        "on_step_start",
                        "generate",
                        {"query": question, "contexts_count": len(contexts)},
                        {},
                    )
                    generation_started = True
                elif kind == "token":
                    if first_token_at is None and value:
                        first_token_at = time.monotonic()
                        await run_blocking(self._run_hook, "on_first_token", first_token_at - start)
                    tokens.append(value)
                elif kind == "citations":
                    answer = "".join(tokens)
                    prompt = _get_default_system_prompt().format(
                        context=generator._format_context(contexts)
                    )
                    usage = {
                        "prompt_tokens": _count_tokens(f"{prompt}\n{question}", model),
                        "completion_tokens": _count_tokens(answer, model),
                    }
                    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
                    cost = self._tracer._pricing.get_cost(
                        provider, model, usage["prompt_tokens"], usage["completion_tokens"]
                    )
                    await run_blocking(
                        self._run_hook,
                        "on_generation_llm_call",
                        model=model,
                        provider=provider,
                        prompt=prompt,
                        query=question,
                        response=answer,
                        usage=usage,
                        cost=cost,
                    )
                    await run_blocking(
                        self._run_hook,
                        "on_step_end",
                        "generate",
                        {"answer": answer},
                        time.monotonic() - start,
                    )
                    generation_started = False
                    await run_blocking(
                        self._run_hook,
                        "on_query_end",
                        answer,
                        value,
                        time.monotonic() - start,
                        {
                            "prompt": usage["prompt_tokens"],
                            "completion": usage["completion_tokens"],
                            "total": usage["total_tokens"],
                        },
                        cost,
                    )
                yield (kind, value)
        except Exception as exc:
            if generation_started:
                await run_blocking(self._run_hook, "on_step_error", "generate", exc)
            await run_blocking(self._run_hook, "on_query_error", exc)
            raise
        finally:
            executor.shutdown(wait=False)

    def stats(self) -> dict[str, Any]:
        return self._pipeline.stats()

    def reset(self) -> None:
        self._run_hook("on_step_start", "reset", {}, {})
        start = time.monotonic()
        try:
            self._pipeline.reset()
            self._run_hook("on_step_end", "reset", {}, time.monotonic() - start)
        except Exception as exc:
            self._run_hook("on_step_error", "reset", exc)
            raise

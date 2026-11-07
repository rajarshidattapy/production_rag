"""Names, placeholder contracts, and built-in versions of runtime prompts."""

from __future__ import annotations

from dataclasses import dataclass

from src.prompt_management import PromptStore, get_prompt_store, resolve_prompt

DEFAULT_SYSTEM = "generation.default_system"
CONDENSE_SYSTEM = "generation.condense_system"
VERIFIER_JUDGE = "healing.verifier_judge"
QUERY_REWRITE = "healing.query_rewrite"
GENERATION_REPAIR = "healing.generation_repair"
CITATION_REPAIR = "healing.citation_repair"
CONTEXT_REPAIR = "healing.context_repair"

PROMPT_FIELDS: dict[str, tuple[str, ...]] = {
    DEFAULT_SYSTEM: ("context",),
    CONDENSE_SYSTEM: (),
    VERIFIER_JUDGE: ("answer", "context", "question"),
    QUERY_REWRITE: (
        "diagnosis",
        "missing_terms",
        "previous",
        "question",
        "question_type",
        "strategy",
    ),
    GENERATION_REPAIR: ("context", "feedback", "num_sources"),
    CITATION_REPAIR: ("context", "feedback", "num_sources"),
    CONTEXT_REPAIR: ("context", "feedback", "num_sources"),
}


@dataclass(frozen=True)
class PromptSpec:
    name: str
    prompt: str
    required_fields: tuple[str, ...]


def builtin_prompt_specs() -> dict[str, PromptSpec]:
    # Local imports avoid a cycle: the modules themselves call ``resolve_builtin``.
    from src.generation.generator import CONDENSE_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT
    from src.healing.prompts import (
        CITATION_REPAIR_PROMPT,
        CONTEXT_REPAIR_PROMPT,
        GENERATION_REPAIR_PROMPT,
        QUERY_REWRITE_PROMPT,
        VERIFIER_JUDGE_PROMPT,
    )

    prompts = {
        DEFAULT_SYSTEM: DEFAULT_SYSTEM_PROMPT,
        CONDENSE_SYSTEM: CONDENSE_SYSTEM_PROMPT,
        VERIFIER_JUDGE: VERIFIER_JUDGE_PROMPT,
        QUERY_REWRITE: QUERY_REWRITE_PROMPT,
        GENERATION_REPAIR: GENERATION_REPAIR_PROMPT,
        CITATION_REPAIR: CITATION_REPAIR_PROMPT,
        CONTEXT_REPAIR: CONTEXT_REPAIR_PROMPT,
    }
    return {
        name: PromptSpec(name=name, prompt=prompt, required_fields=PROMPT_FIELDS[name])
        for name, prompt in prompts.items()
    }


def resolve_builtin(name: str, fallback: str) -> str:
    return resolve_prompt(name, fallback, PROMPT_FIELDS[name])


def sync_builtin_prompts(
    *, actor: str = "application", store: PromptStore | None = None
) -> dict[str, str]:
    target = store or get_prompt_store()
    return {
        name: target.register(name, spec.prompt, spec.required_fields, actor=actor)
        for name, spec in builtin_prompt_specs().items()
    }

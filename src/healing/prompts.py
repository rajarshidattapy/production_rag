"""Prompt templates for the verifier judge, query rewriter, and generation repair.

Repair prompts keep exactly one ``{context}`` placeholder because
``Generator.generate`` formats the system prompt with ``.format(context=...)``;
any other dynamic text is pre-inserted with braces escaped (see ``fill``).
"""

from __future__ import annotations

VERIFIER_JUDGE_PROMPT = """You are a strict verifier for a retrieval-augmented QA system.

Given a QUESTION, numbered CONTEXT passages, and an ANSWER, assess:
- faithfulness: fraction of the answer's factual claims that are directly supported by the context (1.0 = every claim supported).
- relevance: how directly and completely the answer addresses the question (1.0 = fully answers it).
- context_sufficient: true only if the context contains the information needed to answer the question.
- contradiction_detected: true only if context passages contradict each other on a point relevant to the question AND the answer does not explicitly acknowledge that conflict.
- unsupported_claims: short quotes of claims in the answer NOT supported by the context.

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}

Respond with a single JSON object and nothing else:
{{"faithfulness": <0.0-1.0>, "relevance": <0.0-1.0>, "context_sufficient": <true|false>, "contradiction_detected": <true|false>, "unsupported_claims": [<strings>], "explanation": "<one sentence>"}}
"""

QUERY_REWRITE_PROMPT = """You rewrite search queries for a document retrieval system (BM25 keyword search + embedding search).

The previous retrieval attempt did not find context that answers the question.

Original question: {question}
Previous queries tried (do NOT repeat these): {previous}
Question type: {question_type}
Key terms not found in retrieved passages: {missing_terms}
Diagnosis: {diagnosis}

Strategy for this rewrite: {strategy}

Respond with a single JSON object and nothing else:
{{"rewritten_query": "<the new search query>", "rationale": "<one short sentence>"}}
"""

REWRITE_STRATEGIES: dict[str, str] = {
    "expand": (
        "Expand the query: spell out acronyms, add close synonyms and the technical terms a "
        "document on this topic would likely use. Keep it a single focused search query."
    ),
    "keywords": (
        "Reformulate as a compact keyword query (5-10 terms) built from the core concepts, "
        "using different wording than the previous queries."
    ),
    "decompose": (
        "Identify the most important sub-question needed to answer the original question and "
        "write a query for that specific sub-question, using different vocabulary."
    ),
}

_REPAIR_HEADER = """You are a careful research assistant. A previous answer to this question FAILED automated verification.

Why it failed:
{feedback}

Answer the user's question again using ONLY the numbered context below.

**Context:**
{context}

**Hard rules:**
1. Make ONLY claims that are directly supported by the context. If a detail is not in the context, leave it out.
2. Cite every factual sentence with its source number in brackets, e.g. [1] or [2][3]. Valid source numbers are 1 to {num_sources}; never cite a number outside that range.
3. Do NOT use external knowledge.
4. If the context does not contain the answer, reply exactly: "I cannot find sufficient information in the provided documents to answer this question."
5. Respond in the same language as the user's question. Do not translate technical names.
"""

GENERATION_REPAIR_PROMPT = _REPAIR_HEADER

CITATION_REPAIR_PROMPT = (
    _REPAIR_HEADER
    + """6. Pay special attention to citations: each bracketed number must point to the passage that actually contains the cited fact.
"""
)

CONTEXT_REPAIR_PROMPT = (
    _REPAIR_HEADER
    + """6. The passages may disagree with each other. Where they conflict, state the conflict explicitly and cite each side; do not silently pick one.
"""
)


def _escape(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


def fill(template: str, **values: object) -> str:
    """Fill every placeholder except ``{context}``, escaping braces in the values.

    The result is safe to pass to ``Generator.generate(system_prompt=...)``,
    which will substitute ``{context}`` itself.
    """
    escaped = {k: _escape(str(v)) for k, v in values.items()}
    escaped["context"] = "{context}"
    return template.format(**escaped)

"""Screening document text before it reaches the index: credentials, personal data, injection.

Everything here is pattern-based, so it is fast and predictable, but it misses what the
patterns don't describe and can match harmless text. Treat its flags as signals, not verdicts.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable

# Bump when the rules change, so chunks screened under older rules can be found.
RULES_VERSION = 1

# ---- hidden text ------------------------------------------------------------------------
# Unicode "tag" characters render as nothing but can spell out a whole hidden instruction
# (ASCII smuggling), so a run of them is replaced with a visible marker.
_TAG_RUN_RE = re.compile("[\U000e0000-\U000e007f]+")
HIDDEN_TEXT_MARKER = "[removed hidden text]"
# Bidirectional overrides reorder how text displays, and zero-width spaces split words so
# filters miss them. Joiners (U+200C, U+200D) and direction marks are left alone, because
# Persian, Indic scripts, and emoji need them.
_INVISIBLE_RE = re.compile("[\u200b\u2060-\u2064\ufeff\u202a-\u202e\u2066-\u2069]")
# Control tokens of common chat formats, which can make text look like a new turn or role.
_CHAT_MARKUP_RE = re.compile(
    r"<\|(?:im_start|im_end|im_sep|system|user|assistant|endoftext|eot_id|start_header_id|"
    r"end_header_id|begin_of_text)\|>|\[/?INST\]|<</?SYS>>",
    re.IGNORECASE,
)
CHAT_MARKUP_MARKER = "[removed chat markup]"


def neutralize(text: str) -> str:
    """Remove hidden characters and chat-format control tokens. Always safe to apply."""
    text = _TAG_RUN_RE.sub(HIDDEN_TEXT_MARKER, text)
    text = _INVISIBLE_RE.sub("", text)
    return _CHAT_MARKUP_RE.sub(CHAT_MARKUP_MARKER, text)


# ---- prompt injection ---------------------------------------------------------------------
_INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "ignore_instructions",
        re.compile(
            r"\b(?:ignore|disregard|forget|override|bypass)\b[^.\n]{0,40}?"
            r"\b(?:previous|prior|above|earlier|all|any|your|the|these|those)\b[^.\n]{0,20}?"
            r"\b(?:instructions?|prompts?|rules|directions|guidelines)\b"
            # German and Spanish, the other languages the pipeline routes.
            r"|\b(?:ignorier\w*|vergiss)\b[^.\n]{0,40}?\b(?:anweisungen|regeln|vorgaben)\b"
            r"|\b(?:ignora\w*|olvida\w*)\b[^.\n]{0,40}?\b(?:instrucciones|reglas|indicaciones)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "role_override",
        re.compile(
            r"\byou are now\b|\bfrom now on,? you (?:are|will|must|should)\b"
            r"|\bnew (?:system )?instructions\s*:|\bdu bist (?:jetzt|nun)\b|\bahora eres\b",
            re.IGNORECASE,
        ),
    ),
    (
        "prompt_exfiltration",
        re.compile(
            r"\b(?:reveal|print|show|repeat|output|leak|disclose)\b[^.\n]{0,30}?"
            r"\b(?:system prompt|hidden instructions|your instructions|initial prompt)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "conceal_from_user",
        re.compile(
            r"\b(?:do not|don't|never)\s+(?:tell|inform|mention to|reveal to|show)\s+the user\b",
            re.IGNORECASE,
        ),
    ),
    ("chat_markup", re.compile(re.escape(CHAT_MARKUP_MARKER))),
    ("hidden_text", re.compile(re.escape(HIDDEN_TEXT_MARKER))),
)


def injection_signals(text: str) -> list[str]:
    """Names of instruction-like patterns in ``text``. Expects text already ``neutralize``d."""
    return [name for name, pattern in _INJECTION_PATTERNS if pattern.search(text)]


# ---- credentials and personal data ----------------------------------------------------------
_DOTTED_QUAD_RE = re.compile(r"(?:\d{1,3}\.){3}\d{1,3}")


def _luhn_ok(match: re.Match[str]) -> bool:
    digits = [int(d) for d in re.sub(r"\D", "", match.group())]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def _iban_ok(match: re.Match[str]) -> bool:
    iban = match.group().replace(" ", "").upper()
    if not 15 <= len(iban) <= 34:
        return False
    moved = iban[4:] + iban[:4]
    return int("".join(str(int(c, 36)) for c in moved)) % 97 == 1


def _phone_ok(match: re.Match[str]) -> bool:
    # Page ranges, dates, and short numbers share the shape; real phone numbers are longer.
    # A dotted quad is an IP address, which has its own rule.
    if _DOTTED_QUAD_RE.fullmatch(match.group()):
        return False
    return 9 <= len(re.sub(r"\D", "", match.group())) <= 15


def _ssn_ok(match: re.Match[str]) -> bool:
    area, group, serial = match.group().split("-")
    return area not in ("000", "666") and area[0] != "9" and group != "00" and serial != "0000"


def _ip_ok(match: re.Match[str]) -> bool:
    return all(int(part) <= 255 for part in match.group().split("."))


Rule = tuple[str, re.Pattern[str], Callable[[re.Match[str]], bool] | None]

# Credentials come first, so a key isn't half-matched as a phone number.
_CREDENTIAL_RULES: tuple[Rule, ...] = (
    (
        "private_key",
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
        None,
    ),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), None),
    ("api_key", re.compile(r"\bsk-(?:ant-|proj-|or-)?[A-Za-z0-9_-]{20,}"), None),
    ("github_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_\w{30,})"), None),
    ("slack_token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"), None),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}"), None),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"), None),
    ("url_password", re.compile(r"(?<=://)[^\s:/@]+:[^\s@/]+(?=@)"), None),
    (
        "password",
        # Only the value is replaced, so "password: [REDACTED PASSWORD]" still reads naturally.
        # The name may have a prefix, as in db_password or OPENAI_API_KEY.
        re.compile(
            r"\b\w*?(?:password|passwd|pwd|secret|api[_-]?key|access[_-]?token)\s*[:=]\s*"
            r"(?P<value>[^\s\[]\S{5,})",
            re.IGNORECASE,
        ),
        None,
    ),
)
_PII_RULES: tuple[Rule, ...] = (
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), None),
    (
        "iban",
        re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?\b"),
        _iban_ok,
    ),
    ("card_number", re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)"), _luhn_ok),
    ("ssn", re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"), _ssn_ok),
    (
        "phone",
        re.compile(
            r"(?<![\w+])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{1,4}\)[\s.-]?)?\d{2,4}[\s.-]\d{3,8}"
            r"(?:[\s.-]\d{2,4}){0,2}(?![\w])"
        ),
        _phone_ok,
    ),
    ("ip_address", re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])"), _ip_ok),
)


def _apply(text: str, rules: tuple[Rule, ...], replace: bool) -> tuple[str, Counter[str]]:
    found: Counter[str] = Counter()
    for name, pattern, check in rules:

        def swap(match: re.Match[str], _name: str = name, _check: Callable | None = check) -> str:
            if _check is not None and not _check(match):
                return match.group()
            found[_name] += 1
            if not replace:
                return match.group()
            marker = f"[REDACTED {_name.upper()}]"
            if "value" in match.re.groupindex:
                keep = match.start("value") - match.start()
                return match.group()[:keep] + marker
            return marker

        text = pattern.sub(swap, text)
    return text, found


def redact(text: str, *, credentials: bool, pii: bool) -> tuple[str, Counter[str]]:
    """Replace credentials and/or personal data with ``[REDACTED TYPE]`` markers."""
    found: Counter[str] = Counter()
    if credentials:
        text, counts = _apply(text, _CREDENTIAL_RULES, replace=True)
        found.update(counts)
    if pii:
        text, counts = _apply(text, _PII_RULES, replace=True)
        found.update(counts)
    return text, found


def pii_types(text: str) -> list[str]:
    """Kinds of credentials and personal data found in ``text``, without changing it."""
    _, credentials = _apply(text, _CREDENTIAL_RULES, replace=False)
    _, personal = _apply(text, _PII_RULES, replace=False)
    return sorted({*credentials, *personal})

"""Self-healing layer: verify answers, diagnose failures, repair, re-verify, or abstain."""

from src.healing.adaptive_retrieval import AdaptiveRetrievalPolicy
from src.healing.failure_classifier import FailureClassifier
from src.healing.metrics import healing_metrics
from src.healing.query_rewriter import QueryRewriter
from src.healing.state_machine import ABSTENTION_MESSAGE, HealingEngine, route
from src.healing.types import (
    FailureType,
    HealingInfo,
    HealingResult,
    RepairAction,
    VerificationResult,
)
from src.healing.verifier import Verifier

__all__ = [
    "ABSTENTION_MESSAGE",
    "AdaptiveRetrievalPolicy",
    "FailureClassifier",
    "FailureType",
    "HealingEngine",
    "HealingInfo",
    "HealingResult",
    "QueryRewriter",
    "RepairAction",
    "VerificationResult",
    "Verifier",
    "healing_metrics",
    "route",
]

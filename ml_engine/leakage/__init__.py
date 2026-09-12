"""Deterministic leakage and feature-risk analysis.

Semantic (business-meaning) leakage is a separate concern handled by ``ml_engine.reasoning``
after these deterministic checks have run.
"""

from ml_engine.leakage.analyzer import (
    aggregate_feature_risks,
    analyze_leakage,
    default_feature_selection,
)
from ml_engine.leakage.config import LeakageConfig
from ml_engine.leakage.rules import RULES, LeakageContext, build_context

__all__ = [
    "RULES",
    "LeakageConfig",
    "LeakageContext",
    "aggregate_feature_risks",
    "analyze_leakage",
    "build_context",
    "default_feature_selection",
]

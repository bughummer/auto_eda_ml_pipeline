"""What the validator needs to know about columns, and the two ways of learning it.

A proposal is checked twice: when it is offered, where only the EDA report exists, and again
in preparation, where the dataframe itself is loaded. Both checks must reach the same verdict,
so both go through this one description rather than through two type-inspection code paths
that could drift apart.
"""

from dataclasses import dataclass

import pandas as pd

from ml_engine.contracts.common import SemanticType
from ml_engine.contracts.eda import EdaReport
from ml_engine.profiling.config import ProfilingConfig
from ml_engine.profiling.inference import infer_semantic_type

_ARITHMETIC_TYPES = frozenset({SemanticType.NUMERIC_CONTINUOUS, SemanticType.NUMERIC_DISCRETE})


@dataclass(frozen=True, slots=True)
class ColumnTypes:
    """The column facts a derived-feature specification can be checked against."""

    names: frozenset[str]
    arithmetic: frozenset[str]
    date_like: frozenset[str]

    def exists(self, column: str) -> bool:
        return column in self.names

    def is_arithmetic(self, column: str) -> bool:
        """Measured quantities only. A boolean is numeric to pandas but not to arithmetic."""
        return column in self.arithmetic

    def is_date_like(self, column: str) -> bool:
        return column in self.date_like


def column_types_from_frame(
    frame: pd.DataFrame, config: ProfilingConfig | None = None
) -> ColumnTypes:
    """Read the types from the data itself. Used in preparation, where the frame is loaded.

    This runs the profiler's own ``infer_semantic_type`` rather than reading dtypes directly,
    which is what keeps it in step with the EDA path. Dtypes alone disagree: a 0/1 integer
    column is ``int64`` to pandas and a boolean to the profiler, and an identifier is numeric
    to pandas and an identifier to the profiler. Neither is arithmetic.
    """
    config = config or ProfilingConfig()
    row_count = len(frame)
    semantic = {
        str(name): infer_semantic_type(
            frame[name],
            config,
            row_count=row_count,
            unique_count=int(frame[name].nunique(dropna=True)),
            missing_count=int(frame[name].isna().sum()),
        )
        for name in frame.columns
    }
    return _from_semantic_types(semantic)


def column_types_from_eda(eda: EdaReport) -> ColumnTypes:
    """Read the types from the profile. Used when a proposal is offered, before any load.

    The semantic types come from the same inference the frame path uses, so a specification
    accepted here is not refused later on type grounds.
    """
    return _from_semantic_types({c.name: c.semantic_type for c in eda.columns})


def _from_semantic_types(semantic: dict[str, SemanticType]) -> ColumnTypes:
    return ColumnTypes(
        names=frozenset(semantic),
        arithmetic=frozenset(name for name, kind in semantic.items() if kind in _ARITHMETIC_TYPES),
        date_like=frozenset(
            name for name, kind in semantic.items() if kind is SemanticType.DATETIME
        ),
    )

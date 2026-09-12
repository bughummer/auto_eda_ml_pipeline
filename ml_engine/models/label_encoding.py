"""Label encoding wrapper for libraries that only accept integer class labels.

The platform keeps class labels as their original strings everywhere — in metrics, the
confusion matrix, the UI — so a library that demands integers gets the translation here
rather than leaking an encoding convention into the rest of the system.
"""

from typing import Any

import numpy as np
import pandas as pd


class LabelEncodedClassifier:
    """Wraps a classifier, encoding string labels to integers and decoding predictions back."""

    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator
        self.classes_: np.ndarray = np.array([])

    def fit(self, x: pd.DataFrame, y: pd.Series, **kwargs: Any) -> "LabelEncodedClassifier":
        labels = pd.Series(y).astype("string").astype(str)
        self.classes_ = np.array(sorted(labels.unique()))
        lookup = {label: index for index, label in enumerate(self.classes_)}
        self.estimator.fit(x, labels.map(lookup).to_numpy(dtype="int64"), **kwargs)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        encoded = np.asarray(self.estimator.predict(x)).astype(int)
        return self.classes_[encoded]

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.estimator.predict_proba(x))

    @property
    def feature_importances_(self) -> np.ndarray | None:
        return getattr(self.estimator, "feature_importances_", None)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return self.estimator.get_params(deep=deep) if hasattr(self.estimator, "get_params") else {}

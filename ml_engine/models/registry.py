"""The model catalogue.

Plugins register here and nowhere else. Adding a model requires no change to the jobs, the
state machine, the API or the UI.
"""

from ml_engine.contracts.common import ProblemType
from ml_engine.contracts.model import ModelDescriptor
from ml_engine.models.base import ModelPlugin
from ml_engine.models.catboost_plugin import CatBoostClassifierPlugin, CatBoostRegressorPlugin
from ml_engine.models.linear import ElasticNetPlugin, LogisticRegressionPlugin
from ml_engine.models.random_forest import (
    RandomForestClassifierPlugin,
    RandomForestRegressorPlugin,
)
from ml_engine.models.xgboost_plugin import XGBoostClassifierPlugin, XGBoostRegressorPlugin


class UnknownModelError(KeyError):
    """Raised when a configuration references a model that is not registered."""


_REGISTRY: dict[str, ModelPlugin] = {}


def register(plugin: ModelPlugin) -> ModelPlugin:
    if not plugin.name:
        raise ValueError("A model plugin must declare a name")
    if plugin.name in _REGISTRY and type(_REGISTRY[plugin.name]) is not type(plugin):
        raise ValueError(f"Duplicate model plugin name: {plugin.name}")
    _REGISTRY[plugin.name] = plugin
    return plugin


for _plugin in (
    LogisticRegressionPlugin(),
    ElasticNetPlugin(),
    RandomForestClassifierPlugin(),
    RandomForestRegressorPlugin(),
    XGBoostClassifierPlugin(),
    XGBoostRegressorPlugin(),
    CatBoostClassifierPlugin(),
    CatBoostRegressorPlugin(),
):
    register(_plugin)


def get_plugin(name: str) -> ModelPlugin:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise UnknownModelError(
            f"Unknown model {name!r}. Registered models: {', '.join(sorted(_REGISTRY))}."
        ) from exc


def all_plugins() -> list[ModelPlugin]:
    return [_REGISTRY[name] for name in sorted(_REGISTRY)]


def plugins_for(problem_type: ProblemType, *, only_available: bool = True) -> list[ModelPlugin]:
    """Plugins that support this problem type, optionally filtered to installed libraries."""
    plugins = [p for p in all_plugins() if p.supports(problem_type)]
    if only_available:
        plugins = [p for p in plugins if p.is_available()[0]]
    return plugins


def default_model_names(problem_type: ProblemType) -> list[str]:
    """The default model set for a problem type: every supported, installed plugin."""
    return [p.name for p in plugins_for(problem_type)]


def catalogue(problem_type: ProblemType | None = None) -> list[ModelDescriptor]:
    """Descriptors for the API, including unavailable plugins and why they are unavailable."""
    plugins = all_plugins()
    if problem_type is not None:
        plugins = [p for p in plugins if p.supports(problem_type)]
    return [p.descriptor(problem_type) for p in plugins]

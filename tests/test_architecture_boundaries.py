"""The dependency rules from architecture.md, enforced.

If one of these fails, fix the import — do not relax the rule.
"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# ml_engine is the deterministic core. Only its IO boundary may know about AWS.
ALLOWED_AWS_MODULES = {"ml_engine/io/s3.py", "ml_engine/reasoning/client.py"}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def _modules(package: str) -> list[Path]:
    return sorted(p for p in (ROOT / package).rglob("*.py") if "__pycache__" not in str(p))


@pytest.mark.parametrize("module", _modules("ml_engine"), ids=lambda p: str(p.name))
def test_ml_engine_does_not_depend_on_the_platform(module: Path) -> None:
    forbidden = {"fastapi", "backend", "jobs", "starlette", "pydantic_settings"}
    offenders = {i for i in _imports(module) if i.split(".")[0] in forbidden}
    assert not offenders, f"{module.relative_to(ROOT)} imports {offenders}"


@pytest.mark.parametrize("module", _modules("ml_engine"), ids=lambda p: str(p.name))
def test_only_the_io_boundary_touches_aws(module: Path) -> None:
    relative = str(module.relative_to(ROOT))
    if relative in ALLOWED_AWS_MODULES:
        return
    offenders = {i for i in _imports(module) if i.split(".")[0] in {"boto3", "botocore"}}
    assert not offenders, f"{relative} imports {offenders}; AWS access belongs in ml_engine/io"


@pytest.mark.parametrize("module", _modules("jobs"), ids=lambda p: str(p.name))
def test_jobs_do_not_depend_on_the_backend(module: Path) -> None:
    offenders = {i for i in _imports(module) if i.split(".")[0] in {"backend", "fastapi"}}
    assert not offenders, f"{module.relative_to(ROOT)} imports {offenders}"


def test_artifact_paths_are_defined_only_in_the_layout() -> None:
    """No module may hardcode an artifact path; ExperimentLayout owns them."""
    offenders = []
    for package in ("ml_engine", "backend", "jobs"):
        for module in _modules(package):
            if module.name == "layout.py":
                continue
            text = module.read_text(encoding="utf-8")
            for needle in ('"eda/eda.json"', '"comparison/comparison.json"', '"models/"'):
                if needle in text:
                    offenders.append(f"{module.relative_to(ROOT)}: {needle}")
    assert not offenders, f"hardcoded artifact paths: {offenders}"

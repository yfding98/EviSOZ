"""Install a deliberately empty ``src.soz`` package for v13 fit runners."""

from __future__ import annotations

import importlib
import importlib.machinery
from pathlib import Path
import sys
from types import ModuleType


FORBIDDEN_V13_MODULE_PREFIXES = (
    "pandas",
    "src.soz.data",
    "src.soz.concept_run",
    "src.soz.ictal_fit_only_targets_v13",
    "src.soz.ictal_fit_token_view_v13",
    "src.soz.ictal_target_snapshot",
    "src.soz.ictal_recovery_oof",
    "src.soz.ictal_native_eval",
    "src.soz.evidence",
)


def install_v13_minimal_soz_package(project_root: Path) -> bool:
    """Bypass the broad research-package initializer in a clean process.

    Returns ``True`` only when this call installed the minimal package.  Unit
    tests that already imported the broad package may still inspect parsers,
    but a formal runner must start in a clean interpreter and requires the
    returned value to be true before entering ``main``.
    """

    if "src.soz" in sys.modules:
        return False
    root = Path(project_root).resolve()
    package_path = (root / "src" / "soz").resolve()
    if not package_path.is_dir():
        raise RuntimeError("v13 minimal package path is missing")
    src_package = importlib.import_module("src")
    module = ModuleType("src.soz")
    module.__file__ = str(package_path / "__init__.py")
    module.__path__ = [str(package_path)]
    module.__package__ = "src.soz"
    module.__spec__ = importlib.machinery.ModuleSpec(
        "src.soz", loader=None, is_package=True
    )
    sys.modules["src.soz"] = module
    setattr(src_package, "soz", module)
    models_path = package_path / "models"
    models = ModuleType("src.soz.models")
    models.__file__ = str(models_path / "__init__.py")
    models.__path__ = [str(models_path)]
    models.__package__ = "src.soz.models"
    models.__spec__ = importlib.machinery.ModuleSpec(
        "src.soz.models", loader=None, is_package=True
    )
    sys.modules["src.soz.models"] = models
    setattr(module, "models", models)
    assert_forbidden_v13_modules_absent()
    return True


def assert_forbidden_v13_modules_absent() -> None:
    forbidden = tuple(
        name
        for name in sorted(sys.modules)
        if any(name.startswith(prefix) for prefix in FORBIDDEN_V13_MODULE_PREFIXES)
    )
    if forbidden:
        raise RuntimeError(
            "v13 fit runtime imported a forbidden target/data module: "
            + ",".join(forbidden)
        )


__all__ = (
    "FORBIDDEN_V13_MODULE_PREFIXES",
    "assert_forbidden_v13_modules_absent",
    "install_v13_minimal_soz_package",
)

"""Legacy module-path aliasing for pre-0.1.0 MLflow artifacts.

qlib's ``init_instance_by_config`` imports handler classes by the ``module_path``
string baked into the pickled ``task`` artifact. Pre-0.1.0 runs used
``factor_factory.alphas.alpha_158_custom_qlib``; post-migration the canonical
path is ``qlib_strategy_core.handlers.alpha_158_custom``.

``install_finder()`` registers a ``MetaPathFinder`` that transparently maps the
legacy paths to the current ones. Call this once at process startup.

On the training machine qlib_strategy_dev still has physical shim files, so the
finder is purely defensive. On clean inference environments (vnpy node) the
finder is the *only* resolution path for legacy artifacts.
"""

from __future__ import annotations

import importlib
import sys
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from importlib.util import spec_from_loader
from types import ModuleType


LEGACY_TO_CURRENT = {
    "factor_factory.alphas.alpha_158_custom_qlib": "qlib_strategy_core.handlers.alpha_158_custom",
    "factor_factory.alpha_factor_store": "qlib_strategy_core.alpha_factor_store",
}

# Synthetic parent packages so Python's import system can resolve
# "factor_factory.alphas.alpha_158_custom_qlib" without the parents existing on disk.
SYNTHETIC_PARENTS = {
    "factor_factory",
    "factor_factory.alphas",
}


class _AliasLoader(Loader):
    def __init__(self, target_name: str):
        self.target_name = target_name

    def create_module(self, spec):  # noqa: D401
        return importlib.import_module(self.target_name)

    def exec_module(self, module: ModuleType) -> None:  # noqa: D401
        pass  # module already fully initialized by importing target


class _SyntheticPackageLoader(Loader):
    """Creates an empty namespace-like package so child aliases can be found."""

    def create_module(self, spec):  # noqa: D401
        mod = ModuleType(spec.name)
        mod.__path__ = []  # mark as package (enables submodule lookup)
        return mod

    def exec_module(self, module: ModuleType) -> None:  # noqa: D401
        pass


class _LegacyPathFinder(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname in LEGACY_TO_CURRENT:
            return spec_from_loader(fullname, _AliasLoader(LEGACY_TO_CURRENT[fullname]))
        if fullname in SYNTHETIC_PARENTS and fullname not in sys.modules:
            spec = ModuleSpec(fullname, _SyntheticPackageLoader(), is_package=True)
            spec.submodule_search_locations = []
            return spec
        return None


_INSTALLED = False


def install_finder() -> None:
    """Idempotently register the alias MetaPathFinder."""
    global _INSTALLED
    if _INSTALLED:
        return
    sys.meta_path.insert(0, _LegacyPathFinder())
    _INSTALLED = True

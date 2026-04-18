"""qlib factor handler classes.

``module_path`` references baked into MLflow task artifacts point here.
New training runs use ``qlib_strategy_core.handlers.alpha_158_custom``;
legacy runs with ``factor_factory.alphas.alpha_158_custom_qlib`` keep
working via the shim module in qlib_strategy_dev (re-exports from here).
"""

from qlib_strategy_core.handlers.alpha_158_custom import (
    Alpha158Custom,
    Alpha158CustomDataLoader,
)

__all__ = [
    "Alpha158Custom",
    "Alpha158CustomDataLoader",
]

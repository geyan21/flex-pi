"""YAM FlexPi deployment package.

Re-exports are resolved lazily (PEP 562) so that importing this package does
not pull in ``torch``, ``hydra`` and the full ``flexpi`` model stack. The
lightweight modules here -- ``client_rtc_step_broker``, ``server_smoother``,
``server_speed`` -- and their unit tests then run in any environment with
numpy and pytest, which is what their module docstrings promise.

``from experiments.yam.flexpi_policy import YamFlexPiPolicy`` still works and
still imports the heavy stack on first attribute access.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "YamFlexPiPolicy",
    "build_policy_from_checkpoint",
]

_LAZY = {
    "YamFlexPiPolicy": ".deploy_policy",
    "build_policy_from_checkpoint": ".deploy_policy",
}


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module, __name__), name)


def __dir__():
    return sorted(__all__)


if TYPE_CHECKING:  # import-time types for checkers only, never at runtime
    from .deploy_policy import YamFlexPiPolicy, build_policy_from_checkpoint

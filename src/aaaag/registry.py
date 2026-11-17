"""Plugin registries: add a component by adding a file, never by editing a shared list.

    from aaaag.registry import EVIDENCE

    @EVIDENCE.register("ball_proximity", version="1", example={"radius_m": 2.0})
    def ball_proximity(bundle, *, radius_m=2.0):
        ...
        return Evidence("ball_proximity", proba)

Every module under the registry's package is imported on first lookup, so a new file in
src/aaaag/evidence/ is picked up with no other edit -- which is what keeps parallel PRs from
colliding. `version` is part of every cache key: bump it whenever the component's output changes
for the same inputs, or stale cached evidence will be reused. `example` is a parameter set the
contract tests (tests/test_contracts.py) run the component with on a synthetic match, so every
new component is checked against the contracts automatically.

Signatures (enforced by tests/test_contracts.py for every registered component):
    SOURCES   fn(**params) -> MatchBundle
    EVIDENCE  fn(bundle: MatchBundle, **params) -> Evidence
    FUSION    fn(evidence: list[Evidence], meta: MatchMeta, **params) -> np.ndarray (T, 3) float32
    DECODERS  fn(proba: np.ndarray, meta: MatchMeta, **params) -> np.ndarray (T,) int8 labels
    POSTPROCESS  fn(labels: np.ndarray (T,) int8, bundle: MatchBundle, **params) -> np.ndarray (T,) int8
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Component:
    kind: str
    name: str
    version: str
    fn: Callable[..., Any]
    example: dict[str, Any] = field(default_factory=dict)

    @property
    def module(self) -> str:
        return self.fn.__module__


class Registry:
    def __init__(self, kind: str, package: str) -> None:
        self.kind = kind
        self.package = package
        self._items: dict[str, Component] = {}
        self._loaded = False

    def register(
        self, name: str, *, version: str = "1", example: dict[str, Any] | None = None
    ) -> Callable[[Callable], Callable]:
        def decorator(fn: Callable) -> Callable:
            existing = self._items.get(name)
            if existing is not None and existing.fn.__module__ != fn.__module__:
                raise ValueError(
                    f"{self.kind} {name!r} already registered by {existing.module}; pick another name"
                )
            self._items[name] = Component(self.kind, name, str(version), fn, dict(example or {}))
            return fn

        return decorator

    def load(self) -> None:
        """Import every module of the package once, so their decorators run."""
        if self._loaded:
            return
        package = importlib.import_module(self.package)
        for info in pkgutil.walk_packages(package.__path__, prefix=f"{self.package}."):
            importlib.import_module(info.name)
        self._loaded = True

    def get(self, name: str) -> Component:
        self.load()
        if name not in self._items:
            raise KeyError(f"no {self.kind} named {name!r}; available: {', '.join(self.names())}")
        return self._items[name]

    def names(self) -> list[str]:
        self.load()
        return sorted(self._items)


SOURCES = Registry("source", "aaaag.sources")
EVIDENCE = Registry("evidence", "aaaag.evidence")
FUSION = Registry("fusion", "aaaag.fusion")
DECODERS = Registry("decoder", "aaaag.decoders")
POSTPROCESS = Registry("postprocess", "aaaag.postprocess")

ALL = (SOURCES, EVIDENCE, FUSION, DECODERS, POSTPROCESS)

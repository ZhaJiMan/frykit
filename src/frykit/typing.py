from __future__ import annotations

from collections.abc import Callable, Hashable
from pathlib import Path
from typing import Any, TypeAlias, TypeVar

__all__ = ["F", "HashableT", "PathType", "T"]

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])
HashableT = TypeVar("HashableT", bound=Hashable)

PathType: TypeAlias = str | Path

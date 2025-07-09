from __future__ import annotations

from collections.abc import Callable, Hashable
from pathlib import Path
from typing import TypeAlias, TypeVar

T = TypeVar("T")
F = TypeVar("F", bound=Callable)
HashableT = TypeVar("HashableT", bound=Hashable)

PathType: TypeAlias = str | Path

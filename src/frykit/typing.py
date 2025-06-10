from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypeAlias, TypeVar

T = TypeVar("T")
F = TypeVar("F", bound=Callable)

PathType: TypeAlias = str | Path

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")
F = TypeVar("F", bound=Callable)

PathType = str | Path

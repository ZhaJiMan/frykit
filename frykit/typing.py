from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])

PathType = str | Path

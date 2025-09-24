from __future__ import annotations

from collections.abc import Callable, Hashable
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar

if TYPE_CHECKING:
    import numpy as np

__all__ = ["F", "HashableT", "PathType", "RealNumber", "RealNumberT", "T"]

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])
HashableT = TypeVar("HashableT", bound=Hashable)

PathType: TypeAlias = str | Path

RealNumber: TypeAlias = "np.integer | np.floating"
RealNumberT = TypeVar("RealNumberT", bound=RealNumber)

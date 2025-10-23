from __future__ import annotations

from collections.abc import Hashable
from pathlib import Path
from typing import TYPE_CHECKING, ParamSpec, TypeAlias, TypeVar

if TYPE_CHECKING:
    import numpy as np

__all__ = ["HashableT", "P", "PathType", "RealNumber", "RealNumberT", "T"]

T = TypeVar("T")
P = ParamSpec("P")

HashableT = TypeVar("HashableT", bound=Hashable)

PathType: TypeAlias = str | Path

# 赋值操作需要用前向引用
RealNumber: TypeAlias = "np.integer | np.floating"
RealNumberT = TypeVar("RealNumberT", bound=RealNumber)

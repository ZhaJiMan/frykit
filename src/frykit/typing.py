from __future__ import annotations

from collections.abc import Hashable
from os import PathLike
from typing import TYPE_CHECKING, ParamSpec, TypeAlias, TypeVar

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "BytesPath",
    "HashableT",
    "P",
    "RealNumber",
    "RealNumberT",
    "StrOrBytesPath",
    "StrPath",
    "T",
]

T = TypeVar("T")
P = ParamSpec("P")

HashableT = TypeVar("HashableT", bound=Hashable)

StrPath: TypeAlias = str | PathLike[str]
BytesPath: TypeAlias = bytes | PathLike[bytes]
StrOrBytesPath: TypeAlias = StrPath | BytesPath

# 赋值操作需要用前向引用
RealNumber: TypeAlias = "np.integer | np.floating"
RealNumberT = TypeVar("RealNumberT", bound=RealNumber)

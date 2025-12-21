from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from typing import Any, Literal, TypeAlias, TypedDict, cast

from typing_extensions import Unpack

from frykit.utils import format_literal_error, format_type_error

__all__ = ["Config", "ConfigDict", "DataSource", "config"]

DataSource: TypeAlias = Literal["amap", "tianditu"]


def _validate_data_source(value: Any) -> None:
    if value not in {"amap", "tianditu"}:
        raise ValueError(
            format_literal_error("data_source", value, ["amap", "tianditu"])
        )


def _validate_bool(name: str, value: Any) -> None:
    if not isinstance(value, bool):
        raise TypeError(format_type_error(name, value, bool))


class ConfigDict(TypedDict):
    data_source: DataSource
    fast_transform: bool
    skip_outside: bool
    strict_clip: bool


# TODO: 如何减少重复定义
class PartialConfigDict(TypedDict, total=False):
    data_source: DataSource
    fast_transform: bool
    skip_outside: bool
    strict_clip: bool


# TODO: 线程安全
@dataclass(kw_only=True)
class Config:
    """表示全局配置的类"""

    data_source: DataSource = "amap"
    fast_transform: bool = True
    skip_outside: bool = True
    strict_clip: bool = False

    def __post_init__(self) -> None:
        self._field_names = {field.name for field in fields(self)}
        for name in self._field_names:
            self._validate(name, getattr(self, name))

    def assert_field(self, name: str) -> None:
        """断言名字是否属于配置字段"""
        if name not in self._field_names:
            raise ValueError(f"不存在的配置：{name}")

    def _validate(self, name: str, value: Any) -> None:
        match name:
            case "data_source":
                _validate_data_source(value)
            case "fast_transform" | "skip_outside" | "strict_clip":
                _validate_bool(name, value)

    def validate(self, name: str, value: Any) -> None:
        """校验一条配置"""
        self.assert_field(name)
        self._validate(name, value)

    def __setattr__(self, name: str, value: Any) -> None:
        # 允许设置非字段的属性
        self._validate(name, value)
        super().__setattr__(name, value)

    def to_dict(self) -> ConfigDict:
        """将配置转换为字典"""
        return cast(ConfigDict, asdict(self))

    def update(self, **kwargs: Unpack[PartialConfigDict]) -> None:
        """更新配置"""
        # 校验完再更新，避免校验失败导致部分更新
        for name, value in kwargs.items():
            self.validate(name, value)
        for name, value in kwargs.items():
            super().__setattr__(name, value)

    @contextmanager
    def context(self, **kwargs: Unpack[PartialConfigDict]) -> Iterator[None]:
        """创建可以临时修改配置的上下文"""
        config_dict = self.to_dict()
        try:
            self.update(**kwargs)
            yield
        finally:
            self.update(**config_dict)


config = Config()

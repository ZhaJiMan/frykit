from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal

from frykit.utils import format_literal_error, format_type_error

Validator = Callable[[Any], None]


@dataclass
class OptionItem:
    value: Any
    validator: Validator | None


class Option:
    """表示全局配置的类"""

    def __init__(self) -> None:
        self._data: dict[str, OptionItem] = {}

    def _assert_registered(self, name: str) -> None:
        """确保配置是否注册过"""
        if name not in self._data:
            raise ValueError(f"不存在的配置：{name}")

    def __getitem__(self, name: str) -> Any:
        """获取一条配置"""
        self._assert_registered(name)
        return self._data[name].value

    def get_validator(self, name: str) -> Validator | None:
        """获取一条配置的校验函数"""
        return self._data[name].validator

    def __setitem__(self, name: str, value: Any) -> None:
        """设置一条配置"""
        self._assert_registered(name)
        validator = self.get_validator(name)
        if validator is not None:
            validator(value)
        self._data[name].value = value

    def register_option(
        self, name: str, default: Any, validator: Validator | None = None
    ) -> None:
        """注册一条配置。可选校验配置值的校验函数。"""
        self._data[name] = OptionItem(default, validator)


DataSource = Literal["amap", "tianditu"]


def _validate_data_source(data_source: DataSource) -> None:
    """校验 data_source 是否合法"""
    if data_source not in {"amap", "tianditu"}:
        raise ValueError(
            format_literal_error("data_source", data_source, ["amap", "tianditu"])
        )


def _validate_bool(value: bool) -> None:
    """校验是否为布尔类型"""
    if not isinstance(value, bool):
        raise ValueError(format_type_error("value", value, bool))


# 全局配置
_option = Option()
_option.register_option("data_source", "amap", _validate_data_source)
_option.register_option("fast_transform", True, _validate_bool)
_option.register_option("skip_outside", True, _validate_bool)
_option.register_option("strict_clip", False, _validate_bool)


def get_option(name: str) -> Any:
    """获取一条配置"""
    return _option[name]


# TODO: 线程安全
def set_option(options: dict[str, Any]) -> None:
    """用键值对更新配置"""
    for name, value in options.items():
        _option[name] = value


def resolve_option(name: str, value: Any | None) -> Any:
    """解析配置值。当配置值是 None 时使用默认配置，否则校验后返回原值。"""
    if value is None:
        return _option[name]

    validator = _option.get_validator(name)
    if validator is not None:
        validator(value)

    return value


@contextmanager
def option_context(options: dict[str, Any]) -> Generator[None]:
    """临时在上下文中用键值对更新配置"""
    original_options = {}
    for name in options:
        original_options[name] = get_option(name)

    try:
        set_option(options)
        yield
    finally:
        set_option(original_options)

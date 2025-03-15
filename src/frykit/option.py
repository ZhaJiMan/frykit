from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal

from frykit.utils import format_literal_error, format_type_error

__all__ = [
    "get_option",
    "set_option",
    "validate_option",
    "option_context",
]

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
        if name not in self._data:
            raise ValueError(f"不存在的配置：{name}")

    def __getitem__(self, name: str) -> Any:
        self._assert_registered(name)
        return self._data[name].value

    def validate(self, name: str, value: Any) -> None:
        validator = self._data[name].validator
        if validator is not None:
            validator(value)

    def __setitem__(self, name: str, value: Any) -> None:
        # 要求配置曾注册过，并且能通过校验。
        self._assert_registered(name)
        self.validate(name, value)
        self._data[name].value = value

    def update(self, options: dict[str, Any]) -> None:
        for name, value in options.items():
            self[name] = value

    def to_dict(self) -> dict[str, Any]:
        return {name: item.value for name, item in self._data.items()}

    def __repr__(self) -> str:
        return f"Option({self.to_dict()})"

    def register(
        self, name: str, default: Any, validator: Validator | None = None
    ) -> None:
        self._data[name] = OptionItem(default, validator)


DataSource = Literal["amap", "tianditu"]


def _validate_data_source(data_source: DataSource) -> None:
    if data_source not in {"amap", "tianditu"}:
        raise ValueError(
            format_literal_error("data_source", data_source, ["amap", "tianditu"])
        )


def _validate_bool(value: bool) -> None:
    if not isinstance(value, bool):
        raise ValueError(format_type_error("value", value, bool))


def _init_option(option: Option) -> None:
    option.register("data_source", "amap", _validate_data_source)
    option.register("fast_transform", True, _validate_bool)
    option.register("skip_outside", True, _validate_bool)
    option.register("strict_clip", False, _validate_bool)


_option = Option()
_init_option(_option)


def get_option(name: str) -> Any:
    """获取一条配置"""
    return _option[name]


# TODO: 线程安全
def set_option(options: dict[str, Any]) -> None:
    """用键值对更新配置"""
    _option.update(options)


def validate_option(name: str, value: Any) -> None:
    """校验配置值"""
    _option.validate(name, value)


# TODO: 线程安全
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

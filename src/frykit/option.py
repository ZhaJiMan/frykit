from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from frykit.conf import ConfigDict, config
from frykit.utils import deprecator, simple_deprecator

__all__ = [
    "get_option",
    "get_options",
    "set_option",
    "validate_option",
    "option_context",
]


@simple_deprecator("{name} 已弃用，建议直接访问 frykit.config 对象的属性")
def get_option(name: str) -> Any:
    """获取一条配置"""
    config.assert_field(name)
    return getattr(config, name)


@deprecator(alternative="frykit.config.to_dict")
def get_options() -> ConfigDict:
    """获取所有配置"""
    return config.to_dict()


@simple_deprecator("{name} 已弃用，建议直接修改 frykit.config 对象的属性")
def set_option(options: dict[str, Any]) -> None:
    """用键值对更新配置"""
    for name, value in options.items():
        config.assert_field(name)
        setattr(config, name, value)


@deprecator(alternative="frykit.config.validate")
def validate_option(name: str, value: Any) -> None:
    """校验一条配置"""
    config.validate(name, value)


@deprecator(alternative="frykit.config.context")
@contextmanager
def option_context(options: dict[str, Any]) -> Generator[None]:
    """临时在上下文中用键值对更新配置"""
    with config.context(**options):
        yield

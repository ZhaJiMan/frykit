from collections.abc import Generator, Mapping
from contextlib import contextmanager
from typing import Any

from frykit._config import ConfigDict, config
from frykit.utils import deprecator

__all__ = [
    "get_option",
    "get_options",
    "set_option",
    "validate_option",
    "option_context",
]


@deprecator(alternative="frykit.config.{name}")
def get_option(name: str) -> Any:
    """获取一条配置"""
    config.assert_field(name)
    return getattr(config, name)


@deprecator(alternative="frykit.config.to_dict")
def get_options() -> ConfigDict:
    """获取所有配置"""
    return config.to_dict()


@deprecator(alternative="frykit.config.{name} = {value}")
def set_option(name: str, value: Any) -> None:
    """用键值对更新配置"""
    config.assert_field(name)
    setattr(config, name, value)


@deprecator(alternative="frykit.config.validate")
def validate_option(name: str, value: Any) -> None:
    """校验一条配置"""
    config.validate(name, value)


@deprecator(alternative="frykit.config.context")
@contextmanager
def option_context(options: Mapping[str, Any]) -> Generator[None]:
    """临时在上下文中用键值对更新配置"""
    with config.context():
        config.update(options)
        yield

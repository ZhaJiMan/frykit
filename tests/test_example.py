from importlib import import_module
from pathlib import Path
from unittest import TestCase

from loguru import logger

from frykit.utils import chdir_context, new_dir


class TestExample(TestCase):
    def setUp(self) -> None:
        new_dir("image")

    def test_run(self) -> None:
        """运行所有示例脚本

        脚本在 example 目录，通过切换工作目录让图片输出到 image 目录。
        注意 import_module 只会按 sys.path 的内容去查找模块，不受切换工作目录的影响。
        """
        names = [filepath.stem for filepath in Path("example").glob("*.py")]
        with chdir_context("image"):
            for name in names:
                if name not in {"clabel", "river", "nerv_style"}:
                    try:
                        import_module(f"example.{name}")
                        logger.info(name)
                    except Exception:
                        logger.exception(name)

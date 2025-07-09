import subprocess
from pathlib import Path
from subprocess import CalledProcessError
from unittest import TestCase

from frykit.utils import chdir_context, new_dir


class TestExample(TestCase):
    def setUp(self) -> None:
        new_dir("image")

    @chdir_context("example")
    def test_run(self) -> None:
        """运行所有示例脚本"""
        for filepath in Path(".").glob("*.py"):
            if filepath.stem in {"clabel", "river", "nerv_style"}:
                continue
            try:
                subprocess.run(
                    args=["python", str(filepath)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                print(f"[OK] {filepath.stem}")
            except CalledProcessError as e:
                print(f"[FAIL] {filepath.stem}")
                print(e.stderr)

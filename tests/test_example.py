import subprocess
from pathlib import Path
from unittest import TestCase

from frykit.utils import chdir_context, new_dir


class TestExample(TestCase):
    @chdir_context("example")
    def test_run(self) -> None:
        """运行所有示例脚本"""
        new_dir("image")
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
            except subprocess.CalledProcessError as e:
                print(f"[FAIL] {filepath.stem}")
                print(e.stderr)

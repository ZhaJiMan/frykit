'''运行example目录下的所有脚本.'''

# TODO: 换用pytest
# TODO: 加入Mercator和rhumb line的例子.
import subprocess
from pathlib import Path

code_filepath = Path(__file__).absolute()
example_dirpath = code_filepath.parent.parent / 'example'
for example_filepath in example_dirpath.iterdir():
    subprocess.run(['python', str(example_filepath)])
    print(f'> {example_filepath.name}')

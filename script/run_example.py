'''运行example目录下的所有脚本'''

# TODO: 换用pytest
import subprocess
from pathlib import Path

code_filepath = Path(__file__).absolute()
example_dirpath = code_filepath.parent.parent / 'example'
for filepath in example_dirpath.iterdir():
    if filepath.stem == 'clabel':
        continue
    subprocess.run(['python', str(filepath)])
    print(f'> {filepath.name}')

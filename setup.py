from pathlib import Path

import setuptools

root_dirpath = Path(__file__).parent

init_filepath = root_dirpath / 'frykit' / '__init__.py'
with open(str(init_filepath), encoding='utf-8') as f:
    for line in f:
        if line.startswith('__version__'):
            version = line.split('\'')[1]
            break

readme_filepath = root_dirpath / 'README.md'
with open(str(readme_filepath), encoding='utf-8') as f:
    long_description = f.read()

requirements_filepath = root_dirpath / 'requirements.txt'
with open(str(readme_filepath), encoding='utf-8') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name='frykit',
    version=version,
    author='ZhaJiMan',
    author_email='915023793@qq.com',
    description='A simple toolbox for Matplotib and Cartopy',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ZhaJiMan/frykit',
    include_package_data=True,
    packages=setuptools.find_packages(),
    install_requires=requirements,
    python_requires='>=3.9',
)

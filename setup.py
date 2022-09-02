from pathlib import Path
import setuptools

filepath_readme = Path(__file__).parent / 'README.md'
with open(str(filepath_readme), 'r', encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name='frykit',
    version='0.1.3',
    author='ZhaJiMan',
    author_email='915023793@qq.com',
    description='A simple toolbox for Matplotib and Cartopy',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ZhaJiMan/frykit',
    include_package_data=True,
    packages=setuptools.find_packages(),
    # install_requires=['cartopy>=0.20.0']
)
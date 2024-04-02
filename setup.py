from pathlib import Path
import setuptools

readme_filepath = Path(__file__).parent / 'README.md'
with open(str(readme_filepath), 'r', encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name='frykit',
    version='0.5.0',
    author='ZhaJiMan',
    author_email='915023793@qq.com',
    description='A simple toolbox for Matplotib and Cartopy',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ZhaJiMan/frykit',
    include_package_data=True,
    packages=setuptools.find_packages(),
    install_requires=['cartopy>=0.20.0', 'pandas>=1.2.0'],
    python_requires='>=3.9',
)

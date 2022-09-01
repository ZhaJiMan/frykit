from pathlib import Path
import setuptools

setuptools.setup(
    name='frykit',
    version='0.1.3',
    author='ZhaJiMan',
    author_email='915023793@qq.com',
    description='A simple toolbox for Python plotting',
    url='https://github.com/ZhaJiMan/frykit',
    include_package_data=True,
    packages=setuptools.find_packages(),
    install_requires=['cartopy>=0.20.0']
)

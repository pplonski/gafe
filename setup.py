from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
#with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
#    long_description = f.read()

setup(
    name='gafe',
    version='0.0.1',
    description='Genetic Algorithm Feature Engineering',
    #long_description=long_description,
    url='https://github.com/pplonski/gafe',
    author='Piotr Plonski',
    author_email='contact@mljar.com',
    license='Apache-2.0',
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=['scipy', 'pandas', 'numpy', 'sklearn']
)

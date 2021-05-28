from setuptools import find_packages, setup


with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='ml_in_prod_hw1',
    author='made',
    license='MIT',
    install_requires = required
)

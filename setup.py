from setuptools import setup, find_packages

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='rbfln',
   version='0.1',
   description='Radial Basis Functional Link Network',
   license="GPLv3",
   long_description=long_description,
   author='Francisco C. Trucco <franciscoctrucco@gmail.com>, Juan M. Scavuzzo <juansca1229@gmail.com>',
   packages=find_packages(),
)

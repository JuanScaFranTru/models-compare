from setuptools import setup, find_packages

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='models_compare',
   version='0.1',
   description='Compare some Machine Learning models to predict oviposition',
   license="GPLv3",
   long_description=long_description,
   author='Francisco C. Trucco <franciscoctrucco@gmail.com>, Juan M. Scavuzzo <juansca1229@gmail.com>',
   packages=find_packages(),
)

from setuptools import setup,find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="colorectal_cancer_research",
    versision= "0.1",
    author= "Chris Morris",
    packages=find_packages(),
    install_requires = requirements,

)
from setuptools import setup
from setuptools import find_packages

setup(name='BCOT',
      install_requires=[
            "POT",
            "scipy",
            "scikit-learn",
            "setuptools"
      ],
      package_data={'BCOT': ['README.md']},
      packages=find_packages())

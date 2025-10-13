import os
import sys
from setuptools import setup, find_packages

# Check if Python version is supported
if sys.version_info[:2] != (3, 6):
    sys.exit("iPRESTO requires Python 3.6")


here = os.path.abspath(os.path.dirname(__file__))
version = {}
with open(os.path.join(here, "ipresto", "__version__.py")) as f:
    exec(f.read(), version)

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="iPRESTO",
    version=version["__version__"],
    author="Annette Lien",
    author_email="a.lien@posteo.de",
    description="Detection of biosynthetic sub-clusters",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://git.wageningenur.nl/bioinformatics/iPRESTO",
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
    ],
    test_suite="tests",
    python_requires='==3.6.*',
    install_requires=[
          'biopython',
          'pyhmmer',
          'matplotlib',
          'networkx',
          'numpy',
          'gensim==3.8.3',
          'smart_open<5.0.0', # for compatibility with LdaMulticore in gensim 3.8.3
          'pyLDAvis',
          'pandas',
          'scipy',
          'seaborn',
          'statsmodels',
          'sympy',
          'scikit-learn',
      ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov"
        ]
    },
    entry_points={
        "console_scripts": [
            "ipresto-cli = ipresto.cli:main"
        ]
    },
)

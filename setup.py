from pathlib import Path
import sys
from setuptools import setup, find_packages

# Check if Python version is supported
if sys.version_info[:2] != (3, 12):
    sys.exit("subclue requires Python 3.12")

# Get version from __version__.py file
version = {}
version_file = Path(__file__).parent / "subclue" / "__version__.py"
with open(version_file) as f:
    exec(f.read(), version)

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="subclue",
    version=version["__version__"],
    author="Annette Lien",
    author_email="a.lien@posteo.de",
    description="Detection of BGC subcluster motifs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.12,<3.13',
    install_requires=[
          'biopython',
          'rdkit'
      ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov"
        ]
    },
    entry_points={
        "console_scripts": [
            "subclue = subclue.cli:main"
        ]
    },
)

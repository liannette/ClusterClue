## Installation

```bash
# Create a new conda environment with python 3.12
conda create -n clusterclue python=3.12

# Activate the new environment
conda activate clusterclue

# install clusterclue and dependencies
pip install -e .[dev]

# If HMMER is not installed on your system already, install it via conda
conda install -c bioconda hmmer
```
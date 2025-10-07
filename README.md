## Installation

```bash

# Download the HMM profiles for the detection of protein domains
wget https://zenodo.org/records/7006969/files/Pfam_100subs_tc.hmm

# Index the HMM profiles
hmmpress Pfam_100subs_tc.hmm 

# Download this git repository
git clone https://github.com/liannette/ClusterClue.git

# Create a new conda environment with python 3.12 and hmmer
conda create -n clusterclue -c bioconda python=3.12 hmmer=3.4  

# Activate the environment
conda activate clusterclue

# Install ClusterClue and dependencies
pip install ClusterClue/
```


## Example

```bash
clusterclue \
    --out output \
    --motifs input/cc_motifs_240909.txt \
    --gbk input/test_gbks \
    --hmm Pfam_100subs_tc.hmm \
    --cores 10 \
    --verbose \
    > log.txt 2>&1
```
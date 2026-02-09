# ClusterClue

ClusterClue is a Python package for prediction of novel biosynthetic sub-clusters.
It detects co-occurring genes within BGCs and represents them as SubCluster Motifs, which are gene weight matrices that quantify the importance of individual genes. These SubCluster Motifs can be applied to any BGC, enabling systematic detection and linking of similar sub-clusters across BGCs.


## Installation

You can install ClusterClue using either the standard workflow (for SubCluster Motif detection) or the extended setup (for SubCluster Motif building).

### 1. Create a new environment

```bash
# Create a new conda environment with Python 3.12
conda create -n clusterclue python=3.12
conda activate clusterclue
```

### 2. Clone the repository

```bash
git clone https://github.com/liannette/ClusterClue.git
cd ClusterClue
```

### 3. Choose one of the installation modes
a. To detect pregenerated SubCluster Motifs

```bash
pip install .
```

b. To build your own SubCluster Motifs

```bash
pip install ".[build]"
```

(The [build] extra includes additional dependencies required for SubCluster Motif generation.)

### Optional: Install visualization support
If you want to visualize results, also install SubSketch:

```bash
git clone https://github.com/liannette/SubSketch.git
pip install ./SubSketch/
```

## Usage

```bash
# run ClusterClue directly from the commandline to detect SubCluster Motifs
clusterclue detect \
    --out output \
    --gbks input_bgcs/ \
    --hmm biosynthetic_subdomains.hmm \
    --gwms subcluster_motifs.txt \
    --cores 1 \
    --verbose 
```

Full options (clusterclue detect -h):

```text
--gwms <file>           Input file containing gene weight matrices (GWMs) of subcluster motifs.
--out <dir>             Output directory, this will contain all output data files.
--gbks <dir>            Input directory containing gbk files of the gene clusters.
--hmm <file>            Path to the HMM file containing protein domain HMMs that has been processed with hmmpress.
--max_domain_overlap <float>
                        Specify at which overlap percentage domains are considered to overlap. 
                        Domain with the best score is kept (default=0.1).
--visualize_hits        Visualize detected motif hits.
--compound_smiles <file>
                        Path to a TSV file containing bgc_id, compound_name, compound_smiles. 
                        If provided, compound structures will be visualized in the html reports.
-c <int>, --cores <int> Set the number of cores the script may use (default: use all available cores)
-v, --verbose           Prints more detailed information.
```
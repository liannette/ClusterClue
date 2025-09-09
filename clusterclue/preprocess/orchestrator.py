import random
from pathlib import Path
from typing import List, Optional

from clusterclue.preprocess.utils import write_clusters
from clusterclue.preprocess.tokenize.orchestrator import TokenizeOrchestrator
from clusterclue.preprocess.domain_filtering import perform_domain_filtering


def run_preprocess(
    out_dir_path: str,
    existing_clusterfile: Optional[str],
    gbks_file: str,
    gbks_dir_path: str,
    hmm_file_path: str,
    exclude_name: List[str],
    include_contig_edge_clusters: bool,
    max_domain_overlap: float,
    cores: int,
    verbose: bool,
) -> str:
    """Orchestrate all preprocessing steps.

    Preprocess BGCs by tokenizing, filtering non-biosynthetic domains, and removing similar clusters.

    Args:
        out_dir_path (str): Output directory path.
        existing_clusterfile (Optional[str]): Path to an existing cluster file to use instead of tokenizing.
        gbks_file (str): Path to a file for storing the paths to the input GenBank files.
        gbks_dir_path (str): Directory path containing GenBank files.
        hmm_file_path (str): Path to HMM file.
        exclude_name (List[str]): If any string in this list occurs in the gbk filename, this file will not be used for the analysis.
        include_contig_edge_clusters (bool): Whether to include clusters that lie on a contig edge.
        max_domain_overlap (float): If two domains overlap more than this value, only the domain with the highest score is kept.
        cores (int): Number of cores to use.
        verbose (bool): Whether to print verbose output.

    Returns:
        str: Path to the final preprocessed clusters file.
    """
    # Set random seed for reproducibility
    random.seed(595)

    # Create output directory
    out_dir = Path(out_dir_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Tokenize the genes of the clusters into protein domain combinations
    clusters_file_path = out_dir / "clusters_all_domains.csv"
    gene_counts_file_path = out_dir / "clusters_all_domains_gene_counts.txt"
    if clusters_file_path.is_file():
        if verbose:
            print(
                "\nSkipping tokenisation step, because the file"
                f" already exists: {clusters_file_path}"
            )
    elif existing_clusterfile:
        if verbose:
            print(
                f"\nUsing provided file of tokenized BGCs: {existing_clusterfile}."
            )
        write_clusters(clusters, clusters_file_path)
    else:
        if verbose:
            print("\nTokenizing the BGC genes into protein domain combinations")
        TokenizeOrchestrator().run(
            clusters_file_path,
            gene_counts_file_path,
            gbks_file,
            gbks_dir_path,
            hmm_file_path,
            exclude_name,
            include_contig_edge_clusters,
            max_domain_overlap,
            cores,
            verbose,
        )

    # Step 2: Filter non-biosynthetic protein domains
    out_file_path = out_dir / "clusters_biosyn_domains.csv"
    if out_file_path.is_file():
        if verbose:
            print(
                f"\nSkipping domain filtering, because the file already exists: {out_file_path}"
            )
    else:
        domain_filtering_file_path = (
            Path(__file__).parent.parent.parent
            / "data"
            / "biosynthetic_domains.txt"
        )
        counts_file_path = out_dir / "clusters_biosyn_domains_gene_counts.txt"
        perform_domain_filtering(
            clusters_file_path,
            domain_filtering_file_path,
            out_file_path,
            counts_file_path,
            cores,
            verbose,
        )
    clusters_file_path = out_file_path
    
    return clusters_file_path
import random
from pathlib import Path
from typing import List, Optional

from subclue.preprocess.utils import write_clusters
from subclue.preprocess.tokenize.orchestrator import TokenizeOrchestrator
from subclue.preprocess.domain_filtering import perform_domain_filtering


def run_preprocess(
    out_dir_path: str,
    gbks_file: str,
    gbks_dir_path: str,
    hmm_file_path: str,
    exclude_name: List[str],
    include_contig_edge_clusters: bool,
    cores: int,
    verbose: bool,
) -> str:
    """Orchestrate all preprocessing steps.

    Preprocess BGCs by tokenizing, filtering non-biosynthetic domains, and removing similar clusters.

    Args:
        out_dir_path (str): Output directory path.
        gbks_file (str): Path to a file for storing the paths to the input GenBank files.
        gbks_dir_path (str): Directory path containing GenBank files.
        hmm_file_path (str): Path to HMM file.
        exclude_name (List[str]): If any string in this list occurs in the gbk filename, this file will not be used for the analysis.
        include_contig_edge_clusters (bool): Whether to include clusters that lie on a contig edge.
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

    # Create the file containing all protein domains
    all_domains_file = out_dir / "domain_hits_all.txt"
    if all_domains_file.is_file():
        if verbose:
            print(
                "\nSkipping tokenisation step, because the file"
                f" already exists: {all_domains_file}"
            )
    else:
        if verbose:
            print("\nTokenizing the BGC genes into protein domain combinations")
        TokenizeOrchestrator().run(
            all_domains_file,
            gbks_file,
            gbks_dir_path,
            hmm_file_path,
            exclude_name,
            include_contig_edge_clusters,
            cores,
            verbose,
        )

    # # Step 2: Filter non-biosynthetic protein domains
    # out_file_path = out_dir / "clusters_biosyn_domains.csv"
    # if out_file_path.is_file():
    #     if verbose:
    #         print(
    #             f"\nSkipping domain filtering, because the file already exists: {out_file_path}"
    #         )
    # else:
    #     domain_filtering_file_path = (
    #         Path(__file__).parent.parent.parent
    #         / "data"
    #         / "biosynthetic_domains.txt"
    #     )
    #     counts_file_path = out_dir / "clusters_biosyn_domains_gene_counts.txt"
    #     perform_domain_filtering(
    #         clusters_file_path,
    #         domain_filtering_file_path,
    #         out_file_path,
    #         counts_file_path,
    #         cores,
    #         verbose,
    #     )
    # clusters_file_path = out_file_path
    
    return clusters_file_path
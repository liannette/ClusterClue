import re

from collections import Counter
from typing import Tuple, Set
from multiprocessing import Pool
from functools import partial

from clusterclue.preprocess.utils import (
    read_clusters,
    write_clusters,
    read_txt,
)

def extract_domain_base_name(domain: str) -> str:
    """Extract the base name of a domain.

    This function extracts the base name of a protein_domain,
    excluding any '_c' suffix of subPfams.

    Args:
        domain (str): The domain name.

    Returns:
        str: The base name of the domain.
    """
    return re.sub(r"_c\d+$", "", domain)


def filter_domains_in_gene(gene: Tuple[str], include_domains: Set[str]) -> Tuple[str]:
    """Filter domains in a gene based on the include_list.

    This function iterates through the domains in a given gene and filters
    them based on whether their base name (excluding any '_c' suffix) is
    present in the include_list. If a domain matches the criteria, it is
    included in the filtered_gene list.

    Args:
        gene (list of str): A list of domain names in the gene.
        include_domains (set of str): A set of domain base names to include.

    Returns:
        list of str: A list of filtered domain names. If no domains match
            the criteria, a list containing a single tuple with a dash ('-')
            is returned.
    """
    filtered_gene = []
    for domain in gene:
        domain_name = extract_domain_base_name(domain)
        if domain_name in include_domains:
            filtered_gene.append(domain)
    return tuple(filtered_gene) if filtered_gene else ("-",)


def filter_domains_in_cluster(cluster, include_domains, verbose):
    cluster_id, genes = cluster
    filtered_genes = [filter_domains_in_gene(gene, include_domains) for gene in genes]
    return cluster_id, filtered_genes


def perform_domain_filtering(
    in_file_path: str,
    domain_filtering_file_path: str,
    out_file_path: str,
    counts_file_path: str,
    cores: int,
    verbose: bool,
) -> str:
    """
    Wrapper for domain filtering of clusters.

    Args:
        in_file_path (str): Path to the input file containing tokenised clusters.
        domain_filtering_file_path (str): Path to the file containing the list of domains to include.
        out_file_path (str): Path to the output file for writing the domain-filtered clusters.
        counts_file_path (str): Path to the file for writing gene counts.
        verbose (bool): If True, print verbose output.
    """
    if verbose:
        print(f"\nPerforming domain filtering on {in_file_path}")
        print(f"Removing protein domains not listed in {domain_filtering_file_path}")

    # Read the input files
    include_domains = set(read_txt(domain_filtering_file_path))
    clusters = read_clusters(in_file_path)

    # Process each cluster in parallel
    with Pool(cores, maxtasksperchild=1000) as pool:
        process_func = partial(
            filter_domains_in_cluster,
            include_domains=include_domains,
            verbose=verbose,
        )
        results = pool.map(process_func, clusters.items())
    filtered_clusters = {cluster_id: genes for (cluster_id, genes) in results}

    # Write the results to the output files
    write_clusters(filtered_clusters, out_file_path)

    if verbose:
        print(f"\nPerformed domain filtering on {len(clusters)} tokenised clusters.")
        print(f"The domain-filtered clusters have been saved to {out_file_path}")

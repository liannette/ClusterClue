import os
from collections import Counter
from glob import iglob
from multiprocessing import Pool

from Bio import SearchIO
from pathlib import Path
from functools import partial

from ipresto.preprocess.utils import (
    count_non_empty_genes,
    format_cluster_to_string,
    write_gene_counts,
)


def calculate_overlap(tup1, tup2):
    """
    Calculates the overlap between two ranges.

    Args:
        tup1 (tuple): A tuple of two ints, start and end of the first alignment (0-indexed).
        tup2 (tuple): A tuple of two ints, start and end of the second alignment (0-indexed).

    Returns:
        int: The overlap between the two ranges.
    """
    start1, end1 = tup1
    start2, end2 = tup2

    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    return overlap_end - overlap_start


def domains_are_overlapping(tup1, tup2, max_overlap):
    """
    Returns true if there is an overlap between two ranges higher than cutoff.

    Args:
        tup1 (tuple): A tuple of two ints, start and end of the first alignment (0-indexed).
        tup2 (tuple): A tuple of two ints, start and end of the second alignment (0-indexed).
        max_overlap (float): Fraction that two alignments are allowed to overlap.

    Returns:
        bool: True if there is an overlap higher than the cutoff, False otherwise.
    """
    length1 = tup1[1] - tup1[0]
    length2 = tup2[1] - tup2[0]

    overlap = calculate_overlap(tup1, tup2)
    return overlap > min(length1, length2) * max_overlap


def domtable_to_tokenized_cluster(domtable_path, max_domain_overlap):
    """Parses a domtab file and extracts domain information.

    Args:
        domtable_path (str): Path to the domtab file.
        max_domain_overlap (float): Max overlap allowed between two domains before they are
            considered overlapping.

    Returns:
        list: A list of lists where each sublist represents a gene and contains tuples
            of domains.
    """
    queries = SearchIO.parse(domtable_path, "hmmscan3-domtab")
    cds_before = 0
    cds_num = 0
    total_genes = 0

    # list of lists for the domains in the cluster where each sublist is a gene
    tokenized_genes = []
    all_domain_hits = []

    while True:
        query = next(queries, None)
        if query is None:
            # end of queries/queries is empty
            break
        # for every cds that has a hit
        dom_matches = []
        q_id = query.id
        # make sure that bgcs with _ in name do not get split
        bgc, q_id = q_id.split("_gid")
        q_id = q_id.split("_")
        cds_num, total_genes = map(int, q_id[-1].split("/"))
        sum_info = [q.split(":")[-1] for q in q_id[:-1]]
        # for every hit in each cds
        for hit in query:
            match = hit[0]
            domain = match.hit_id
            range_q = match.query_range
            bitsc = match.bitscore
            dom_matches.append((domain, range_q, bitsc))
        dels = []
        if len(query) > 1:
            for i in range(len(query) - 1):
                for j in range(i + 1, len(query)):
                    # if there is a significant overlap delete the one with
                    # the lower bitscore
                    if domains_are_overlapping(
                        dom_matches[i][1], dom_matches[j][1], max_domain_overlap
                    ):
                        if dom_matches[i][2] >= dom_matches[j][2]:
                            dels.append(j)
                        else:
                            dels.append(i)
        cds_matches = [dom_matches[i] for i in range(len(query)) if i not in dels]
        cds_matches.sort(key=lambda x: x[1][0])

        # Add to all_domain_hits
        for domain, range_q, bitscore in cds_matches:
            all_domain_hits.append(
                [
                    bgc,
                    sum_info,
                    cds_num,
                    total_genes,
                    domain,
                    range_q,
                    bitscore,
                ]
            )

        # Collect domain matches for the current CDS
        cds_doms = tuple(domain for domain, _, _ in cds_matches)

        # If a CDS has no domains, add '-' to indicate gap
        gene_gap = cds_num - cds_before - 1
        if gene_gap > 0:
            gaps = [("-",) for _ in range(gene_gap)]
            tokenized_genes += gaps

        tokenized_genes.append(cds_doms)
        cds_before = cds_num

    # Add gaps to the end of the cluster if there are any
    end_gap = total_genes - cds_num
    if end_gap > 0:
        gaps = [("-",) for _ in range(end_gap)]
        tokenized_genes += gaps

    cluster_id = Path(domtable_path).stem
    return cluster_id, tokenized_genes, all_domain_hits


def process_domtable(
    domtable_path: str, max_domain_overlap: float, verbose: bool
) -> list:
    """
    Processes a single domtable file and returns the tokenized cluster.

    Args:
        domtable_path (str): Path to the domtable file.
        max_overlap (float): Max overlap allowed between two domains before they are considered overlapping.
        verbose (bool): If True, prints additional information during processing.

    Returns:
        list: A list of lists where each sublist represents a gene and contains tuples of domains.
              Returns None if there is an error in processing.
    """
    try:
        return domtable_to_tokenized_cluster(domtable_path, max_domain_overlap)
    except Exception as e:
        if verbose:
            cluster_id = Path(domtable_path).stem
            print(f"  excluding {cluster_id}, error in processing : {e}")
        return None


def filter_non_empty_genes(clusters, min_genes, verbose):
    """
    Filters the clusters for non-empty genes.

    Args:
        clusters (list): List of clusters from processing domtables.
        min_genes (int): Minimum number of genes with domain hits required per cluster.
        verbose (bool): If True, prints additional information during filtering.

    Returns:
        list: A list containing the filtered clusters
    """
    filtered_clusters = []

    for cluster in clusters:
        cluster_id, tokenized_genes, _ = cluster
        n_genes_with_domains = count_non_empty_genes(tokenized_genes)

        if n_genes_with_domains >= min_genes:
            filtered_clusters.append(cluster)
        elif verbose:
            print(f"  excluding {cluster_id}, only {n_genes_with_domains} genes with domain hits (min {min_genes})")

    return sorted(filtered_clusters, key=lambda x: x[0])


def write_summary_header(summary_file):
    summary_header = [
        "bgc",
        "g_id",
        "p_id",
        "location",
        "orf_num",
        "tot_orf",
        "domain",
        "q_range",
        "bitscore",
    ]
    summary_header = "\t".join(summary_header)
    summary_file.write(f"{summary_header}\n")


def format_summary_line(
    cluster_id, sum_info, cds_num, total_genes, domain, range_q, bitscore
):
    return "{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
        cluster_id,
        "\t".join(sum_info),
        cds_num,
        total_genes,
        domain,
        ";".join(map(str, range_q)),
        bitscore,
    )


def process_domtables(
    domtables_dir_path,
    cluster_file_path,
    gene_counts_file_path,
    domain_hits_file_path,
    min_genes,
    max_domain_overlap,
    cores,
    verbose,
):
    """
    Processes the domtables in a directory and writes the clusters to a file.

    Args:
        domtables_dir_path (str): Path to the directory containing the domtables.
        cluster_file_path (str): Path to the output file where the clusters will be written.
        gene_counts_file_path (str): Path to the output file where the gene counts will be written.
        domain_hits_file_path (str): Path to the output file where all domain matches will be written.
        min_genes (int): Minimum number of genes required in a cluster.
        domain_overlap_cutoff (float): Minimum overlap required between two domains to be considered overlapping.
        cores (int): Number of CPU cores to use for parallel processing.
        verbose (bool): If True, prints additional information during processing.

    Raises:
        IOError: If the cluster file or summary file cannot be written.

    Notes:
        - The function writes the clusters to a file and prints a message if a cluster
          is excluded due to having fewer genes than the minimum required.
        - The function writes the summary of domain matches to a file.
        - The function skips domtable files that cannot be parsed or have no domain hits.
        - The function counts the number of genes in each cluster and writes this information
          to a separate file with the same name as the cluster file, but with "_gene_counts.txt" appended.
    """
    if verbose:
        print("\nParsing domtables into tokenized clusters...")
        
    domtable_paths = list(iglob(os.path.join(domtables_dir_path, "*.domtable")))

    # Process each domtable in parallel
    with Pool(cores, maxtasksperchild=1000) as pool:
        process_func = partial(
            process_domtable,
            max_domain_overlap=max_domain_overlap,
            verbose=verbose,
        )
        results = pool.map(process_func, domtable_paths)
    clusters = [res for res in results if res is not None]
    clusters.sort(key=lambda x: x[0])  # Sort clusters by cluster_id
    n_failed = len(domtable_paths) - len(clusters)

    # Filter the clusters for non-empty genes
    clusters = filter_non_empty_genes(clusters, min_genes, verbose)
    n_converted = len(clusters)
    n_excluded = len(domtable_paths) - len(clusters) - n_failed

    cluster_file = open(cluster_file_path, "w")
    for cluster_id, tokenized_genes, _ in clusters:
        cluster_file.write(format_cluster_to_string(cluster_id, tokenized_genes))
    cluster_file.close()


    # Write the gene counts to a file
    gene_counter = Counter()
    for _, tokenized_genes, _ in clusters:
        gene_counter.update(tokenized_genes)
    write_gene_counts(gene_counter, gene_counts_file_path)

    # Write the summary file
    domain_hits_file = open(domain_hits_file_path, "w")
    write_summary_header(domain_hits_file)
    for _, _, domain_hits in clusters:
        for domain_hit in domain_hits:
            domain_hits_file.write(format_summary_line(*domain_hit))
    domain_hits_file.close()

    if verbose:
        print(f"\nProcessed {len(domtable_paths)} domtables:")
        print(f" - {n_converted} domtables were converted to tokenised clusters.")
        print(f" - {n_excluded} excluded for having < {min_genes} non-empty genes")
        print(f" - {n_failed} domtables failed be converted to tokenised clusters")

import logging
from collections import Counter
from multiprocessing import Pool
from Bio import SearchIO
from pathlib import Path
from functools import partial
from clusterclue.utils import worker_init
from clusterclue.clusters.utils import (
    count_non_empty_genes,
    format_cluster_to_string,
    write_gene_counts,
)

logger = logging.getLogger(__name__)


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


def extract_query_metadata(query_id):
    """
    Extracts BGC name and CDS metadata from a query ID.
    
    Args:
        query_id (str): Query ID in format "bgcname_gidX_Y_Z_N/M"
        
    Returns:
        tuple: (bgc_name, summary_info, cds_number, total_genes)
    """
    bgc_name, gid_part = query_id.split("_gid")
    id_parts = gid_part.split("_")
    
    # Extract CDS number and total genes from last part (format: "N/M")
    cds_num, total_genes = map(int, id_parts[-1].split("/"))
    
    # Extract summary info from remaining parts
    summary_info = [part.split(":")[-1] for part in id_parts[:-1]]
    
    return bgc_name, summary_info, cds_num, total_genes


def collect_domain_hits(query):
    """
    Collects domain hits from a query and returns them sorted by position.
    
    Args:
        query: SearchIO query object containing domain hits
        
    Returns:
        list: List of tuples (domain_name, query_range, bitscore) sorted by start position
    """
    domain_hits = [
        (hit[0].hit_id, hit[0].query_range, hit[0].bitscore)
        for hit in query
    ]
    # Sort by start position for efficient overlap detection
    domain_hits.sort(key=lambda x: x[1][0])
    return domain_hits


def remove_overlapping_domains(domain_hits, max_overlap):
    """
    Removes overlapping domains, keeping the one with higher bitscore.
    Uses an optimized algorithm that takes advantage of sorted domain positions.
    
    Args:
        domain_hits (list): List of (domain, range, bitscore) tuples, sorted by start position
        max_overlap (float): Maximum allowed overlap fraction
        
    Returns:
        list: Filtered list with overlapping domains removed
    """
    if len(domain_hits) <= 1:
        return domain_hits
    
    domains_to_delete = set()
    
    for i in range(len(domain_hits) - 1):
        # Skip if this domain is already marked for deletion
        if i in domains_to_delete:
            continue
            
        current_range = domain_hits[i][1]
        current_score = domain_hits[i][2]
        
        # Check potential overlaps with subsequent domains
        for j in range(i + 1, len(domain_hits)):
            if j in domains_to_delete:
                continue
            
            next_range = domain_hits[j][1]
            next_score = domain_hits[j][2]
            
            # Early termination: if next domain starts after current ends, no overlap possible
            if next_range[0] >= current_range[1]:
                break
            
            # Check if domains overlap beyond threshold
            if domains_are_overlapping(current_range, next_range, max_overlap):
                # Keep the domain with higher bitscore
                if current_score >= next_score:
                    domains_to_delete.add(j)
                else:
                    domains_to_delete.add(i)
                    break  # Current domain is deleted, move to next
    
    # Return filtered list
    return [domain_hits[i] for i in range(len(domain_hits)) if i not in domains_to_delete]


def create_tokenized_genes(queries, max_domain_overlap):
    """
    Processes all queries and creates tokenized gene representation with gaps.
    
    Args:
        queries: Iterator of SearchIO query objects
        max_domain_overlap (float): Max overlap allowed between domains
        
    Returns:
        tuple: (tokenized_genes, all_domain_hits, bgc_name)
            - tokenized_genes: List of gene tuples, with ("-",) for genes without domains
            - all_domain_hits: List of all domain hit information
            - bgc_name: Name of the BGC
    """
    tokenized_genes = []
    all_domain_hits = []
    previous_cds_num = 0
    total_genes = 0
    bgc_name = None
    empty_gene = ("-",)
    
    for query in queries:
        # Extract metadata from query ID
        bgc_name, summary_info, cds_num, total_genes = extract_query_metadata(query.id)
        
        # Collect and process domain hits for this gene
        domain_hits = collect_domain_hits(query)
        filtered_domains = remove_overlapping_domains(domain_hits, max_domain_overlap)
        
        # Store detailed domain hit information
        for domain, range_q, bitscore in filtered_domains:
            all_domain_hits.append([
                bgc_name,
                summary_info,
                cds_num,
                total_genes,
                domain,
                range_q,
                bitscore,
            ])
        
        # Add gap genes (genes without domain hits between previous and current CDS)
        gap_size = cds_num - previous_cds_num - 1
        if gap_size > 0:
            tokenized_genes.extend([empty_gene] * gap_size)
        
        # Add current gene's domains
        gene_domains = tuple(domain for domain, _, _ in filtered_domains)
        tokenized_genes.append(gene_domains)
        previous_cds_num = cds_num
    
    # Add gap genes at the end if needed
    end_gap_size = total_genes - previous_cds_num
    if end_gap_size > 0:
        tokenized_genes.extend([empty_gene] * end_gap_size)
    
    return tokenized_genes, all_domain_hits, bgc_name


def process_domtable(domtable_path, max_domain_overlap):
    """
    Parses a domtab file and extracts domain information.
    
    Args:
        domtable_path (str): Path to the domtab file.
        max_domain_overlap (float): Max overlap allowed between two domains before they are
            considered overlapping.

    Returns:
        tuple: (cluster_id, tokenized_genes, all_domain_hits)
            - cluster_id: Identifier for the cluster (from filename)
            - tokenized_genes: List of gene tuples with domain information
            - all_domain_hits: List of detailed domain hit information
    """
    queries = SearchIO.parse(domtable_path, "hmmscan3-domtab")
    
    tokenized_genes, all_domain_hits, bgc_name = create_tokenized_genes(
        queries, max_domain_overlap
    )
    
    cluster_id = Path(domtable_path).stem
    return cluster_id, tokenized_genes, all_domain_hits


def process_domtable_with_error_handling(
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
        return process_domtable(domtable_path, max_domain_overlap)
    except Exception as e:
        logger.error(f"Error while processing {Path(domtable_path).stem}: {e}")
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
        else:
            logger.debug(
                f"Excluding {cluster_id}, only {n_genes_with_domains} genes with domain hits (min {min_genes})"
            )

    return sorted(filtered_clusters, key=lambda x: x[0])


def write_summary_header(summary_file):
    """Writes the header row for the domain hits summary file."""
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
    summary_file.write("\t".join(summary_header) + "\n")


def format_summary_line(cluster_id, sum_info, cds_num, total_genes, domain, range_q, bitscore):
    """Formats a single domain hit as a tab-separated line."""
    sum_info_str = "\t".join(sum_info)
    range_str = ";".join(map(str, range_q))
    return f"{cluster_id}\t{sum_info_str}\t{cds_num}\t{total_genes}\t{domain}\t{range_str}\t{bitscore}\n"


def process_domtables(
    domtables_dir_path,
    cluster_file_path,
    gene_counts_file_path,
    domain_hits_file_path,
    min_genes,
    max_domain_overlap,
    cores,
    verbose,
    log_queue,
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
        log_queue (multiprocessing.Queue): Queue for logging in multiprocessing.

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
    logger.info("Parsing domtables into tokenized clusters...")

    domtable_paths = list(Path(domtables_dir_path).glob("*.domtable"))

    # Process each domtable in parallel
    pool = Pool(
        cores, maxtasksperchild=100, initializer=worker_init, initargs=(log_queue,)
    )
    with pool:
        process_func = partial(
            process_domtable_with_error_handling,
            max_domain_overlap=max_domain_overlap,
            verbose=verbose,
        )
        results = list(pool.imap_unordered(process_func, domtable_paths, chunksize=10))
        
    clusters = [res for res in results if res is not None]
    clusters.sort(key=lambda x: x[0])  # Sort clusters by cluster_id

    # Filter the clusters for non-empty genes
    filtered_clusters = filter_non_empty_genes(clusters, min_genes, verbose)

    # Write the clusters to a file
    with open(cluster_file_path, "w") as cluster_file:
        for cluster_id, tokenized_genes, _ in filtered_clusters:
            cluster_file.write(format_cluster_to_string(cluster_id, tokenized_genes))

    # Write the gene counts to a file
    gene_counter = Counter()
    for _, tokenized_genes, _ in filtered_clusters:
        gene_counter.update(tokenized_genes)
    write_gene_counts(gene_counter, gene_counts_file_path)

    # Write the domain hits summary file (batched for efficiency)
    with open(domain_hits_file_path, "w") as domain_hits_file:
        write_summary_header(domain_hits_file)
        lines = []
        for _, _, domain_hits in filtered_clusters:
            for domain_hit in domain_hits:
                lines.append(format_summary_line(*domain_hit))
        domain_hits_file.writelines(lines)

    # Log summary
    n_failed = len(domtable_paths) - len(clusters)
    n_converted = len(filtered_clusters)
    n_excluded = len(clusters) - len(filtered_clusters)

    summary_parts = []
    if n_converted > 0:
        summary_parts.append(f"{n_converted} converted")
    if n_excluded > 0:
        summary_parts.append(f"{n_excluded} excluded (< {min_genes} genes)")
    if n_failed > 0:
        summary_parts.append(f"{n_failed} failed")
    summary = ", ".join(summary_parts) if summary_parts else "no files processed"
    logger.info(f"Domtable to tokenized BGCs: {len(domtable_paths)} total - {summary}")
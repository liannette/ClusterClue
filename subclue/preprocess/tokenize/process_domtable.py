import os
from collections import Counter
from glob import iglob
from multiprocessing import Pool
from Bio import SearchIO
from pathlib import Path
from functools import partial

from subclue.preprocess.utils import format_cluster_to_string


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


def extract_info_from_query_id(query_id):
    """
    Extracts information from the query ID. 

    Extracts bgc id, and cds info (gene id, protein id, location), cds num and num of total genes in BGC

    Args:
        query_id (str): The query ID string. Example: BGC0000001_gid:orfP_pid:AEK75490.1_loc:0;1083;+_1/29

    Returns:
        tuple: A tuple containing the extracted information.
    """
    # Extract bgc name, gene id, protein id, location, and cds info
    # Example query.id: BGC0000001_gid:orfP_pid:AEK75490.1_loc:0;1083;+_1/29
    # Make sure that bgcs with _ in name do not get split
    bgc = query_id.split("_gid:")[0]
    # split the string into each of the other fields
    query_info = query_id[len(bgc)+1:].split("_")
    genome_id = query_info[0].split(":")[1]
    protein_id = query_info[1].split(":")[1]
    location = query_info[2].split(":")[1]
    cds_num, total_genes = map(int, query_info[3].split("/"))
    return bgc, [genome_id, protein_id, location], cds_num, total_genes


def domtable_to_tokenized_cluster(domtable_path):
    """Parses a domtab file and extracts domain information.

    Args:
        domtable_path (str): Path to the domtab file.

    Returns:
        list: A list of lists where each sublist represents a gene and contains tuples
            of domains.
    """
    queries = SearchIO.parse(domtable_path, "hmmscan3-domtab")

    all_domain_hits = []

    while True:
        query = next(queries, None)
        if query is None:
            # end of queries/queries is empty
            break
        # Each query represents one CDS with domain hits
        bgc, sum_info, cds_num, total_genes = extract_info_from_query_id(query.id)

        # Get all domains in CDS
        dom_matches = []
        for hit in query:
            domain = hit[0].hit_id          # target name
            range_q = hit[0].query_range    # ali coord: (from-1, to)
            bitsc = hit[0].bitscore         # this domain: score
            dom_matches.append((domain, range_q, bitsc))

        # Add to all_domain_hits
        for domain, range_q, bitscore in dom_matches:
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

    cluster_id = Path(domtable_path).stem
    return cluster_id, all_domain_hits


def process_domtable(
    domtable_path: str, verbose: bool
) -> list:
    """
    Processes a single domtable file and returns the tokenized cluster.

    Args:
        domtable_path (str): Path to the domtable file.
        verbose (bool): If True, prints additional information during processing.

    Returns:
        list: A list of lists where each sublist represents a gene and contains tuples of domains.
              Returns None if there is an error in processing.
    """
    try:
        return domtable_to_tokenized_cluster(domtable_path)
    except Exception as e:
        if verbose:
            cluster_id = Path(domtable_path).stem
            print(f"  excluding {cluster_id}, error in processing : {e}")
        return None


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
    domain_hits_file_path,
    cores,
    verbose,
):
    """
    Processes the domtables in a directory and writes the clusters to a file.

    Args:
        domtables_dir_path (str): Path to the directory containing the domtables.
        domain_hits_file_path (str): Path to the output file where all domain matches will be written.
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
            verbose=verbose,
        )
        results = pool.map(process_func, domtable_paths)
    clusters = [res for res in results if res is not None]
    clusters.sort(key=lambda x: x[0]) # Sort clusters by cluster_id

    if verbose:
        print(f"\nProcessed {len(domtable_paths)} domtables.")
        n_failed = len(domtable_paths) - len(clusters)
        if n_failed > 0:
            print(f"  {n_failed} domtables failed to process or had no domain hits.")

    with open(domain_hits_file_path, "w") as f:
        write_summary_header(f)
        for _, domain_hits in clusters:
            for domain_hit in domain_hits:
                f.write(format_summary_line(*domain_hit))

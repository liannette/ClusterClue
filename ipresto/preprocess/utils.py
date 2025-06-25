from collections import Counter, OrderedDict
from typing import Dict, List, Tuple


def count_non_empty_genes(genes: List[Tuple[str]]) -> int:
    """
    Returns a dictionary with clusters that have at least one non-empty gene.

    Args:
        genes (list): A list of tuples, each tuple representing the domains in a gene.

    Returns:
        int: The number of genes that are not empty.
    """
    return sum(1 for g in genes if g != ("-",))


def parse_cluster_line(line: str) -> Tuple[str, List[Tuple[str]]]:
    """
    Parses a line from a cluster file and returns a tuple with the cluster identifier
    and a list of genes.

    Args:
        line (str): A line from a cluster file.

    Returns:
        tuple: A tuple with the cluster identifier and a list of tuples, each tuple representing
            the domains in a gene.
    """
    line = line.strip().split(",")
    cluster_id = line[0]
    genes = [tuple(gene.split(";")) for gene in line[1:]]
    return cluster_id, genes


def read_clusters(in_file_path: str) -> Dict[str, List[Tuple[str]]]:
    """
    Reads a cluster file and returns a dictionary with cluster identifiers as keys
    and lists of tokensied genes as values.

    Args:
        in_file_path (str): Path to the input cluster file.

    Returns:
        dict: A dictionary with cluster identifiers as keys and lists of tuples, each tuple
            representing the domains in a gene, as values.
    """
    clusters = {}
    with open(in_file_path, "r") as infile:
        for line in infile:
            cluster_id, genes = parse_cluster_line(line)
            clusters[cluster_id] = genes
    return clusters


def format_cluster_to_string(cluster_id: str, genes: List[Tuple[str]]) -> str:
    """
    Formats a cluster as a string.

    Args:
        cluster_id (str): The cluster identifier.
        genes (list): A list of tuples, each tuple representing the domains in a gene.

    Returns:
        str: A formatted string representing the cluster.
    """
    tokenised_genes = [";".join(gene) for gene in genes]
    return f"{cluster_id},{','.join(tokenised_genes)}\n"


def count_gene_occurrences(clusters: Dict[str, List[Tuple[str]]]) -> Counter:
    """
    Counts the occurrences of each gene across all clusters.

    Args:
        clusters (dict): A dictionary with cluster ids as keys and lists of tuples,
            each tuple representing the domains in a gene, as values.

    Returns:
        Counter: A Counter object with genes as keys and their counts as values.
    """
    return Counter(g for genes in clusters.values() for g in genes)


def write_clusters(
    clusters: Dict[str, List[Tuple[str]]], outfile_path: str
) -> None:
    """
    Writes clusters to a specified output file.

    Args:
        clusters (dict): A dictionary with cluster identifiers as keys and lists of tuples,
            each tuple representing the domains in a gene, as values.
        outfile_path (str): The path to the output file where the clusters will be written.

    Raises:
        IOError: If the file cannot be written.
    """
    with open(outfile_path, "w") as f:
        for cluster_id in sorted(clusters.keys()):
            genes = clusters[cluster_id]
            f.write(format_cluster_to_string(cluster_id, genes))


def write_gene_counts(gene_counter: Counter, outfile_path: str) -> None:
    """
    Writes the gene counts from tokenized clusters to a specified output file.

    Args:
        gene_counter (Counter): A Counter object containing gene counts.
        outfile_path (str): The path to the output file where the gene counts will be written.

    Raises:
        IOError: If the file cannot be written.
    """
    total_gene_cnt = sum(gene_counter.values())
    try:
        with open(outfile_path, "w") as f:
            f.write(f"#Total\t{total_gene_cnt}\n")
            for gene, count in gene_counter.most_common():
                f.write(f"{';'.join(gene)}\t{count}\n")
    except IOError as e:
        print(f"Error writing to file {outfile_path}: {e}")
        raise


def write_representatives(
    representative_clusters: Dict[str, List[str]], outfile_path: str
) -> None:
    """
    Writes representative BGCs and their associated BGCs to a file.

    Args:
        representative_clusters (dict): A dictionary where keys are representative BGCs and values
            are lists of associated BGCs, including the representative BGC.
        representatives_file_path (str): The file path where the representatives will be written.

    Returns:
        str: The file path where the representatives were written.
    """
    try:
        with open(outfile_path, "w") as f:
            for repr_cluster, ass_clusters in representative_clusters.items():
                f.write(f"{repr_cluster}\t{','.join(ass_clusters)}\n")
    except IOError as e:
        print(f"Error writing to file {outfile_path}: {e}")
        raise


def read_txt(in_file_path: str) -> List[str]:
    """Reads a text file into a list of strings, stripping whitespace.

    Args:
        in_file (str): Path to the input file.

    Returns:
        list of str: A list of lines from the file, with leading and trailing whitespace removed.
    """
    with open(in_file_path, "r") as f:
        return [line.strip() for line in f]


def remove_empty_clusters(
    clusters: Dict[str, List[Tuple[str]]], 
    min_genes: int, 
    verbose: bool
) -> Dict[str, List[Tuple[str]]]:
    """
    Removes clusters with fewer than `min_genes` genes from the clusters dictionary.

    Parameters:
        clusters (dict): The dictionary of clusters to filter.
        min_genes (int): The minimum number of genes with domains a cluster should have.
        verbose (bool): If True, print additional information during processing.

    Returns:
        dict: The filtered dictionary of clusters, where each cluster has at least `min_genes` genes.

    """
    filtered_clusters = OrderedDict()
    for clus, genes in clusters.items():
        n_genes = count_non_empty_genes(genes)
        if n_genes < min_genes:
            if verbose:
                print(
                    f"  Excluding {clus}: contains only {n_genes} genes with "
                    f"domain hits."
                )
        else:
            filtered_clusters[clus] = genes

    if verbose:
        excluded = len(clusters) - len(filtered_clusters)
        if excluded:
            print(
                f"{excluded} clusters excluded for having less than {min_genes} "
                "genes with domain hits."
            )

    return filtered_clusters


def read_clusters_and_remove_empty(
    in_file_path: str, 
    min_genes: int, 
    verbose: bool
) -> Dict[str, List[Tuple[str]]]:
    """
    Reads clusters from a file and removes those with fewer than `min_genes` genes.

    Parameters:
        in_file_path (str): Path to the input file containing clusters.
        min_genes (int): Minimum number of genes with domains a cluster should have.
        verbose (bool): If True, print additional information during processing.

    Returns:
        dict: A dictionary of clusters with at least `min_genes` non-empty genes.
    """
    clusters = read_clusters(in_file_path)
    return remove_empty_clusters(clusters, min_genes, verbose)
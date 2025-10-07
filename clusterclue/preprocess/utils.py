from typing import Dict, List, Tuple
    

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


def read_txt(in_file_path: str) -> List[str]:
    """Reads a text file into a list of strings, stripping whitespace.

    Args:
        in_file (str): Path to the input file.

    Returns:
        list of str: A list of lines from the file, with leading and trailing whitespace removed.
    """
    with open(in_file_path, "r") as f:
        return [line.strip() for line in f]


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

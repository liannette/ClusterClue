import os
import sys
import random
import time
from collections import Counter, OrderedDict
from functools import partial
from itertools import combinations, islice, chain
from multiprocessing import Pool, Queue
import networkx as nx
import logging
from sympy import binomial as ncr
from clusterclue.utils import worker_init
from clusterclue.clusters.utils import (
    format_cluster_to_string,
    read_clusters,
    write_gene_counts,
    write_representatives,
)

logger = logging.getLogger(__name__)

def generate_adjacent_pairs(domains):
    """Generate unique adjacent domain pairs (unordered), skipping pairs containing '-'."""
    result = set()
    for i in range(len(domains) - 1):
        d1, d2 = domains[i], domains[i + 1]
        if d1 != "-" and d2 != "-":
            # Put in canonical order (cheaper than sort)
            pair = (d1, d2) if d1 < d2 else (d2, d1)
            result.add(pair)
    return result


def calc_adj_index(domains1, domains2):
    """
    Calculate the adjacency index between two clusters.

    The adjacency index is a measure of similarity between two clusters based on
    the adjacency of their domains. Two domains are considered adjacent if they
    are consecutive in the cluster and are not separated by a '-' character.

    Parameters:
    doms1 (list of str): The first cluster, represented as a list of domain strings.
    doms2 (list of str): The second cluster, represented as a list of domain strings.

    Returns:
    float: The adjacency index, a value between 0.0 and 1.0, where 0.0 indicates no
           adjacency and 1.0 indicates perfect adjacency between the two clusters.

    Note:
    - If either cluster has no adjacent domain pairs (i.e., all domains are separated
      by '-'), the function returns 0.0.


    Returns the adjacency index between two clusters

    doms1, doms2: list of str, domainlist of a cluster

    If there is an empty gene between two domains these two domains are not
        adjacent
    """
    domain_pairs1 = generate_adjacent_pairs(domains1)
    domain_pairs2 = generate_adjacent_pairs(domains2)

    # If either cluster has no domain pairs, the adjacency index is 0.0
    if not domain_pairs1 or not domain_pairs2:
        return 0.0

    intersection = domain_pairs1 & domain_pairs2
    if not intersection:
        return 0.0

    # Calculate adjacency index as ratio of intersection to union of domain pairs
    union_size = len(domain_pairs1) + len(domain_pairs2) - len(intersection)
    return len(intersection) / union_size


def is_contained(doms1, doms2):
    """
    Check if all domains from one cluster are contained in the other cluster.

    Parameters:
    doms1 (list of str): Domain list of the first cluster.
    doms2 (list of str): Domain list of the second cluster.

    Returns:
    bool: True if all domains from one cluster are contained in the other, False otherwise.
    """
    set1 = {d for d in doms1 if d != "-"}
    set2 = {d for d in doms2 if d != "-"}
    return set1.issubset(set2) or set2.issubset(set1)


def extract_domains(tokenised_genes):
    """
    Extracts the domains from a list of tokenised genes.

    Args:
        tokenised_genes (list): A list of tokenised genes.

    Returns:
        list: A list of domains.
    """
    return list(chain.from_iterable(tokenised_genes))


def generate_edge(bgc_pair, adj_cutoff):
    """
    Calculate similarity scores between two BGCs and return an edge if above cutoff.

    Args:
        bgc_pair (tuple): A tuple containing two BGCs, where each BGC is represented
                          as a tuple of (bgc_name, genes).
        adj_cutoff (float): The adjacency index cutoff for similarity.

    Returns:
        tuple or None: A tuple (bgc_name1, bgc_name2) if the similarity is above the cutoff
                       or one BGC is contained in the other. Otherwise, returns None.
    """
    (bgc_name1, genes1), (bgc_name2, genes2) = bgc_pair

    # Extract domains from the genes
    domains1 = list(chain.from_iterable(genes1))
    domains2 = list(chain.from_iterable(genes2))

    # Check if the adjacency index exceeds the cutoff
    adj_index = calc_adj_index(domains1, domains2)
    if adj_index > adj_cutoff:
        return bgc_name1, bgc_name2

    # Check if one BGC is contained within the other
    if is_contained(domains1, domains2):
        return bgc_name1, bgc_name2


def determine_chunksize(batch_size, cores):
    """
    Determine the optimal chunksize for parallel processing (~20× chunks per core).

    Args:
        batch_size (int): Total number of tasks to process.
        cores (int): Number of available CPU cores.

    Returns:
        int: The calculated chunksize for efficient parallel processing.
    """
    # Calculate an initial chunksize aiming for ~20× chunks per core
    chunksize = max(batch_size // (cores * 20), 5)
    
    # Ensure that chunksize doesnt exceed maximum allowable size
    return min(chunksize, sys.maxsize)
    

def generate_edges(clusters, cutoff, cores, edges_file, verbose, log_queue):
    """Returns pairs of clusters whose similarity exceeds cutoff, written to edges_file."""
    logger.info("Generating similarity scores")

    # Remove existing temp file
    if os.path.exists(edges_file):
        os.remove(edges_file)

    # Prepare cluster pairs
    total_size = int(ncr(len(clusters), 2))
    batch_size = int(ncr(25000, 2)) # pairs of 25k BGCs is the maximum size
    batch_num = (total_size + batch_size - 1) // batch_size  # round up division
    bgc_pairs = combinations(clusters.items(), 2)

    pool = Pool(
        cores,
        maxtasksperchild=100,
        initializer=worker_init,
        initargs=(log_queue,)
    )
    with open(edges_file, "a") as f, pool:
        for i in range(batch_num):
            start_time = time.time()

            # adjust batch size for the last batch
            if i == batch_num - 1:
                batch_size = total_size - batch_size * (batch_num - 1)

            batch = islice(bgc_pairs, batch_size)
            chunk_size = determine_chunksize(batch_size, cores)
            worker_fn = partial(generate_edge, adj_cutoff=cutoff)

            # Use unordered map for better load balancing
            for edge in pool.imap_unordered(worker_fn, batch, chunksize=chunk_size):
                if edge:
                    line = "\t".join(map(str, edge))
                    f.write(f"{line}\n")

            # log time estimate based on one loop
            if verbose and i == 0:
                elapsed = time.time() - start_time
                t_est = elapsed * batch_num
                hours = int(t_est / 3600)
                minutes = int(t_est % 3600 / 60)
                seconds = int(t_est % 3600 % 60)
                logger.info(f"Estimated time: {hours}h{minutes}m{seconds}s")


            logger.info(f"{i+1}/{batch_num} batches processed")


def generate_graph(edges, verbose):
    """Returns a networkx graph

    edges: list/generator of tuples, (pair1,pair2,{attributes})
    """
    logger.info("Generating graph from edges, this may take a while")

    graph = nx.Graph()
    graph.add_edges_from(edges)
    logger.info(
        f"Generated graph with {graph.number_of_nodes()} nodes "
        f"and {graph.number_of_edges()} edges"
    )
    return graph


def read_edges(file_path):
    """Yields edges from temp file

    file_path: str
    """
    with open(file_path, "r") as inf:
        for line in inf:
            parts = line.strip().split("\t")
            yield parts[0], parts[1]


def find_representatives(cliques, d_l_dict, graph):
    """
    Returns {representative:[clique]} based on cluster/bgc with most domains in clique

    cliques: list of lists of strings, cliques of clusters
    d_l_dict: dict of {clus_name:amount_of_domains(int)}
    graph: networkx graph structure of the cliques
    The longest cluster is chosen (most domains). If there are multiple
        longest clusters then the cluster with the least connections is
        chosen (to preserve most information).
    """
    representative_bgcs = OrderedDict()
    redundant_bgcs = set()

    for cliq in cliques:
        # Filter out already processed BGCs
        cliq = [bgc for bgc in cliq if bgc not in redundant_bgcs]
        if not cliq:
            continue

        # Find the BGC with the maximum number of domains
        domlist = [(bgc, d_l_dict[bgc]) for bgc in cliq]
        maxdoml = max(doml for _, doml in domlist)
        clus_maxlen = [bgc for bgc, doml in domlist if doml == maxdoml]

        # If there are multiple, choose the one with the minimum degree
        if len(clus_maxlen) > 1:
            min_degr = min(graph.degree(bgc) for bgc in clus_maxlen)
            rep = random.choice(
                [bgc for bgc in clus_maxlen if graph.degree(bgc) == min_degr]
            )
        else:
            rep = clus_maxlen[0]

        # Update the representative BGCs
        if rep not in representative_bgcs:
            representative_bgcs[rep] = set()
        representative_bgcs[rep].update(cliq)

        # Mark the remaining BGCs as processed
        cliq.remove(rep)
        redundant_bgcs.update(cliq)

    return representative_bgcs


def find_all_representatives(d_l_dict, g):
    """Iterates find_representatives until there are no similar bgcs

    d_l_dict: dict of {clus_name:amount_of_domains(int)}
    g: networkx graph structure containing the cliques
    all_reps_dict: dict of {representative:[represented]}
    """
    logger.info("Filtering out similar bgcs.")

    all_reps_dict = {}
    subg = g.subgraph(g.nodes)
    i = 1
    while subg.number_of_edges() != 0:
        logger.info(
            f"  iteration {i}, edges (similarities between bgcs) left: {subg.number_of_edges()}"
        )
        cliqs = nx.algorithms.clique.find_cliques(subg)
        # make reproducible by making the cliqs have the same order every time
        # sort first each cliq alphabetically, then cliqs alphabetically,
        # then on length, so longest are first and order is the same
        cliqs = sorted(sorted(cl) for cl in cliqs if len(cl) > 1)
        cliqs.sort(key=len, reverse=True)
        reps_dict = find_representatives(cliqs, d_l_dict, subg)
        subg = subg.subgraph(reps_dict.keys())
        # merge reps_dict with all_reps_dict
        for key, vals in reps_dict.items():
            if key not in all_reps_dict:
                all_reps_dict[key] = vals
            else:
                # merge represented clusters in a new representative
                newvals = []
                for old_rep in vals:
                    # if statement for bgcs already represented by this
                    # representative and thus no longer in all_reps_dict
                    if old_rep in all_reps_dict.keys():
                        newv = [v for v in all_reps_dict[old_rep]]
                        newvals += newv
                        del all_reps_dict[old_rep]
                all_reps_dict[key] = set(newvals)
        i += 1
    return all_reps_dict


def calculate_domain_lengths(tokenised_clusters):
    """
    Calculates the number of domains for each BGC, excluding empty genes ("-").

    Args:
        tokenised_clusters (dict): A dictionary where keys are BGCs and values are lists of domains.

    Returns:
        dict: A dictionary where keys are BGCs and values are the number of domains.
    """
    return {
        bgc: sum(len(g) for g in genes if g != ("-",))
        for bgc, genes in tokenised_clusters.items()
    }


def get_representatives(clusters, graph):
    """
    Identifies representative clusters from the given tokenised clusters.

    Args:
        clusters (dict): A dictionary where keys are BGCs and values are lists of domains.
        graph (nx.Graph): A graph representing the similarity between clusters.
        sim_cutoff (float): Similarity cutoff for generating edges.
        cores (int): Number of cores to use for parallel processing.

    Returns:
        dict: A dictionary where keys are representative BGCs and values are lists of associated BGCs.
    """
    domains_per_bgc = calculate_domain_lengths(clusters)
    representative_bgcs = find_all_representatives(domains_per_bgc, graph)
    unique_bgcs = {bgc: [bgc] for bgc in clusters if bgc not in graph}
    representative_bgcs.update(unique_bgcs)
    return representative_bgcs


def filter_similar_clusters(
    in_file_path: str,
    out_file_path: str,
    counts_file_path: str,
    representatives_file_path: str,
    edges_file_path: str,
    sim_cutoff: float,
    cores: int,
    verbose: bool,
    log_queue: Queue
) -> str:
    """Removes clusters based on similarity and writes the filtered clusters to a file.

    This function reads a file containing tokenised clusters, filters out clusters based on a
    similarity cutoff, and writes the filtered clusters to a new file. If redundancy
    filtering is disabled or if the filtered file already exists, the function skips
    the filtering process.
    """
    logger.info(f"Performing similarity filtering on {in_file_path}.")

    # Read clusters from input file
    clusters = read_clusters(in_file_path)
    # Generate edges between clusters based on similarity
    generate_edges(clusters, sim_cutoff, cores, edges_file_path, verbose, log_queue)
    # Generate a graph from the edges
    graph = generate_graph(read_edges(edges_file_path), verbose)

    # Find representative clusters
    representative_bgcs = get_representatives(clusters, graph)
    write_representatives(representative_bgcs, representatives_file_path)

    # Count the occurence of each gene
    gene_counter = Counter()
    for cluster_id in representative_bgcs:
        genes = clusters[cluster_id]
        gene_counter.update(genes)

    # Write the results to the output files
    write_gene_counts(gene_counter, counts_file_path)
    with open(out_file_path, "w") as outfile:
        for cluster_id in representative_bgcs:
            genes = clusters[cluster_id]
            outfile.write(format_cluster_to_string(cluster_id, genes))

    # Clean up temporary files
    os.remove(edges_file_path)

    logger.info("Similarity filtering complete.")
    logger.info(
            f"Selected {len(representative_bgcs)} representatives for {len(clusters)} clusters"
        )
    logger.info(f"Representative clusters written to {out_file_path}")

"""
kNN graph–based overlapping clustering using Jaccard similarity for sparse gene-module data.

Core ideas:
- Jaccard similarity on sparse gene presence/absence
- kNN graph construction
- Overlapping community detection via Leiden → LPAM

Designed for:
- Hundreds of thousands of modules
- High-dimensional sparse features
- Non-exclusive cluster membership
"""

import logging
import numpy as np
import networkx as nx
from sklearn.preprocessing import MultiLabelBinarizer

from clusterclue.utils import worker_init
from multiprocessing import Pool
from functools import partial

from collections import defaultdict

import os
import time

logger = logging.getLogger(__name__)


def create_sparse_matrix(modules):
    """Convert iterable of gene-sets into a sparse binary matrix."""
    mlb = MultiLabelBinarizer(sparse_output=True)
    X = mlb.fit_transform(modules)
    return X


def jaccard_cutoff_sparse(X, batch_size=5000, jaccard_threshold=0.3):
    """
    Compute kNN graph edges based on Jaccard similarity.

    Args:
        X: n_modules × n_genes sparse binary matrix (csr_matrix)
        batch_size: number of modules to process at once
        jaccard_threshold: minimum Jaccard similarity to keep edge [0,1]

    Returns:
        NetworkX graph with weighted edges (only real similarities)
    """

    n = X.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    X = X.tocsr()

    logger.info(f"Building Jaccard graph (n={n}, threshold={jaccard_threshold})")

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        block = X[start:end]

        # Compute intersection and union efficiently
        # intersection: block @ X.T
        inter = block.dot(X.T).toarray()
        # union: |A| + |B| - |A ∩ B|
        block_sum = block.sum(axis=1).A1[:, None]  # column vector
        X_sum = X.sum(axis=1).A1[None, :]         # row vector
        union = block_sum + X_sum - inter
        union[union == 0] = 1  # avoid division by zero
        sim = inter / union

        # For each module in block, keep top_k neighbors
        for i in range(end - start):
            src = start + i
            high_sim_idx = np.where(sim[i] >= jaccard_threshold)[0]
            for dst in high_sim_idx:
                if src != dst:
                    G.add_edge(src, dst, weight=float(sim[i, dst]))
        if start % 50_000 == 0:
            logger.info(f"Processed {start}/{n} nodes")

    logger.info(f"Jaccard graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def jaccard_overlapping_clustering(modules, jaccard_threshold=0.3, min_comm_size=5, log_queue=None, cores=1):
    """End-to-end pipeline: modules → Jaccard graph → overlapping communities"""
    X = create_sparse_matrix(modules)
    logger.info(f"Sparse matrix shape: {X.shape}")

    jaccard_threshold = 0.5
    # G = jaccard_cutoff_sparse(X, jaccard_threshold=jaccard_threshold)

    # pickle graph for debugging
    import pickle
    from pathlib import Path
    graph_path = Path(f"/home/lien002/nobackup_lien002/projects/clusterclue_publication/asdb5_motifs/output/temp/jaccard_graph_{jaccard_threshold}.pkl")
    if not graph_path.exists():
        X = create_sparse_matrix(modules)
        logger.info(f"Sparse matrix shape: {X.shape}")
        G = jaccard_cutoff_sparse(X, jaccard_threshold=jaccard_threshold)
        with open(graph_path, "wb") as f:
            pickle.dump(G, f)
        logger.info(f"Saved Jaccard graph to {graph_path}")
    else:
        with open(graph_path, "rb") as f:
            G = pickle.load(f)
            logger.info(f"Loaded precomputed Jaccard graph from {graph_path}")

    min_size = min_comm_size

    # Step 1: connected components
    components = [list(c) for c in nx.connected_components(G) if len(c) >= min_size]
    logger.info(f"{len(components)} connected components >= {min_size} nodes")

    sizes = [len(c) for c in components]
    logger.info(f"Component sizes: min={min(sizes)}, max={max(sizes)}, median={np.median(sizes)}")

    # Step 2: parallel clique finding
    all_cliques = parallel_clique_finding(G, components, min_size=min_size, cores=cores, log_queue=log_queue)


    # map back to modules
    module_cliques = [[modules[i] for i in c] for c in all_cliques]

    n_clustered_final = len(set(i for c in all_cliques for i in c))
    logger.info(
        f"Final output: {len(module_cliques)} overlapping communities (min {min_comm_size} nodes each), "
        f"total {n_clustered_final} ({n_clustered_final / len(modules) * 100:.2f}%) nodes."
    )
    
    return module_cliques



def combined_worker_init(G, log_queue=None):
    worker_init(log_queue)
    global _global_G
    _global_G = G


def find_cliques_in_subgraph(component, min_size=5):
    """Find all cliques in the graph G with at least min_size nodes."""
    logger.info(f"PID {os.getpid()}: Finding cliques in subgraph with {len(component)} nodes")
    start_time = time.time()

    G = _global_G.subgraph(component).copy()
    cliques = [c for c in nx.find_cliques(G) if len(c) >= min_size]

    elapsed = time.time() - start_time
    logger.info(f"PID {os.getpid()}: Found {len(cliques)} cliques (min {min_size} nodes each) in subgraph with {len(component)} nodes in {elapsed:.2f} seconds")
    return cliques


def parallel_clique_finding(G, components, min_size=5, cores=1, log_queue=None):
    """Find cliques in each component in parallel."""

    components = components[:20]  # DEBUG: limit to first 20 components
    cores = 10

    logger.info(f"Starting parallel clique finding in {len(components)} components using {cores} cores")
    with Pool(
        processes=cores,
        initializer=combined_worker_init,
        initargs=(G, log_queue),
        maxtasksperchild=10,
    ) as pool:
        func = partial(find_cliques_in_subgraph, min_size=min_size)
        results = pool.map(func, components)

    # Flatten list of lists
    all_cliques = [clique for sublist in results for clique in sublist]
    logger.info(f"Total cliques found: {len(all_cliques)}")
    return all_cliques


def invert_communities(communities):
    """Convert community list into node → communities mapping"""
    membership = defaultdict(list)
    for cid, comm in enumerate(communities):
        for node in comm:
            membership[node].append(cid)
    return membership
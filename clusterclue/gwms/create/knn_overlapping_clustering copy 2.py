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
from collections import defaultdict
import numpy as np
import networkx as nx
from sklearn.preprocessing import MultiLabelBinarizer
from cdlib import algorithms
import igraph as ig
import leidenalg
from clusterclue.utils import worker_init
from multiprocessing import Pool
from functools import partial
import time

import os
from joblib import Parallel, delayed
import multiprocessing

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

    logger.info(f"Jaccard graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def run_leiden(G, resolution=1.0):
    """Run Leiden clustering on a NetworkX graph."""
    nodes = list(G.nodes())
    node2idx = {n: i for i, n in enumerate(nodes)}
    idx2node = {i: n for n, i in node2idx.items()}

    edges = [(node2idx[u], node2idx[v]) for u, v in G.edges()]
    weights = [G[u][v]["weight"] for u, v in G.edges()]

    g = ig.Graph(n=len(nodes), edges=edges, directed=False)
    g.es["weight"] = weights

    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=42,
    )
    node2leiden = {idx2node[i]: cid for i, cid in enumerate(partition.membership)}
    
    # group nodes by Leiden cluster
    leiden2nodes = defaultdict(list)
    for node, cid in node2leiden.items():
        leiden2nodes[cid].append(node)

    return leiden2nodes


def recursive_leiden(
    G,
    resolution=1.0,
    max_cluster_edges=50000,
    min_cluster_size=20,
    max_depth=5,
    depth=0,
):
    """
    Recursively apply Leiden until clusters are small enough.

    Returns:
        List[List[node]]: terminal Leiden clusters
    """
    if depth >= max_depth:
        logger.warning(f"Max recursion depth reached ({depth}); accepting cluster with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return [list(G.nodes())]

    leiden2nodes = run_leiden(G, resolution=resolution)
    logger.info(f"Leiden found {len(leiden2nodes)} clusters at resolution {resolution}")

    terminal_clusters = []
    skip_cnt = 0

    for cid, nodes in leiden2nodes.items():
        if len(nodes) < min_cluster_size:
            skip_cnt += 1
            continue
        
        subG = G.subgraph(nodes).copy()
        edges = subG.number_of_edges()

        if edges <= max_cluster_edges:
            terminal_clusters.append(nodes)
        else:
            logger.info(
                f"Recursing Leiden: depth={depth}, "
                f"cluster={cid}, size={len(nodes)}, edges={edges}, "
                f"resolution={resolution}"
            )
            subG = G.subgraph(nodes).copy()
            terminal_clusters.extend(
                recursive_leiden(
                    subG,
                    resolution=resolution * 1.2,
                    max_cluster_edges=max_cluster_edges,
                    min_cluster_size=min_cluster_size,
                    max_depth=max_depth,
                    depth=depth + 1,
                )
            )
    if skip_cnt > 0:
        logger.info(f"Skipped {skip_cnt} small clusters (<{min_cluster_size} nodes) at depth {depth}")

    return terminal_clusters


def _recursive_leiden(
    G,
    resolution=1.0,
    max_cluster_size=500,
    min_cluster_size=20,
    max_depth=10,
    depth=0,
):
    """
    Recursively apply Leiden until clusters are small enough.

    Returns:
        List[List[node]]: terminal Leiden clusters
    """
    if depth >= max_depth:
        logger.warning(f"Max recursion depth reached ({depth}); accepting cluster with {G.number_of_nodes()} nodes")
        return [list(G.nodes())]

    leiden2nodes = run_leiden(G, resolution=resolution)
    logger.info(f"Leiden found {len(leiden2nodes)} clusters at resolution {resolution}, depth {depth}")

    terminal_clusters = []

    for cid, nodes in leiden2nodes.items():
        if len(nodes) < min_cluster_size:
            continue

        if len(nodes) <= max_cluster_size:
            terminal_clusters.append(nodes)
            continue
        
        subG = G.subgraph(nodes).copy()
        edges = subG.number_of_edges()

        logger.info(
            f"Recursing Leiden: depth={depth}, "
            f"cluster={cid}, size={len(nodes)}, edges={edges}, "
            f"resolution={resolution}"
        )
        subG = G.subgraph(nodes).copy()
        terminal_clusters.extend(
            _recursive_leiden(
                subG,
                resolution=resolution,
                max_cluster_size=max_cluster_size,
                min_cluster_size=min_cluster_size,
                max_depth=max_depth,
                depth=depth + 1,
            )
        )
        
    return terminal_clusters


# def lpam_cluster(nodes, G, k_lpam, min_size=5):
#     """LPAM on single Leiden cluster"""
#     subG = G.subgraph(nodes).copy()
#     sub_comms = algorithms.lpam(subG, k=k_lpam, threshold=0.25, distance='amp')
#     sub_comms = [c for c in sub_comms.communities if len(c) >= min_size]
#     return sub_comms


def lpam_within_leiden(G, leiden_clusters, min_comm_size=5):
    """
    Run LPAM inside each Leiden cluster.
    Returns a list of overlapping communities (global node IDs)
    """
    all_communities = []
    for nodes in leiden_clusters:
        
        subG = G.subgraph(nodes).copy()
        E = subG.number_of_edges()
        ram_gb = 8 * E * E / 1e9
        k_lpam=max(5, int(len(nodes)**0.5))
        logger.info(
            f"Running LPAM on terminal Leiden cluster using k={k_lpam} "
            f"({len(nodes)} nodes, {E:,} edges, ~{ram_gb:.1f} GB RAM required)"
        )
        sub_comms = algorithms.lpam(
            subG,
            k=k_lpam,
            threshold=0.25,
            distance="amp",
            seed=42,
        )
        sub_comms = [c for c in sub_comms.communities if len(c) >= min_comm_size]
        all_communities.extend(sub_comms)

    return all_communities


# def max_edges_per_leiden(max_gb=32):
#     """Max edges before LPAM explodes"""
#     max_bytes = max_gb * (1024 ** 3)
#     max_edges = int(np.sqrt(max_bytes / 8))  # int64
#     return max_edges


def leiden_lpam_clustering(G, leiden_resolution=1.0, min_leiden_size=20, max_leiden_edges=50000, min_comm_size=5):
    """Recursive Leiden → LPAM nested clustering pipeline"""

    # Recursive Leiden to get manageable clusters
    logger.info(f"Starting recursive Leiden clustering (min nodes per cluster {min_leiden_size}, max edges per cluster: {max_leiden_edges:,})")
    leiden_clusters = recursive_leiden(
        G,
        resolution=leiden_resolution,
        max_cluster_edges=max_leiden_edges,
        min_cluster_size=min_leiden_size,
        max_depth=10,
    )
    logger.info(f"Recursive Leiden produced {len(leiden_clusters)} terminal clusters.")

    communities = lpam_within_leiden(
        G,
        leiden_clusters,
        min_comm_size=min_comm_size,
    )
    logger.info(f"LPAM found {len(communities)} overlapping communities (min {min_comm_size} nodes each)")
    return communities


def jaccard_overlapping_clustering(modules, jaccard_threshold=0.3, min_comm_size=5, log_queue=None, cores=1):
    """End-to-end pipeline: modules → Jaccard graph → overlapping communities"""
    # X = create_sparse_matrix(modules)
    # logger.info(f"Sparse matrix shape: {X.shape}")

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

    #max_ram_gb = 16
    #max_edges = max_edges_per_leiden(max_ram_gb)

    leiden_resolution=1.0
    min_leiden_size=20
    max_leiden_size=500

    # logger.info(
    #     f"Starting recursive Leiden clustering "
    #     f"(min nodes {min_leiden_size}, max nodes {max_leiden_size})"
    # )
    # leiden_clusters = _recursive_leiden(
    #     G,
    #     resolution=leiden_resolution,
    #     min_cluster_size=min_leiden_size,
    #     max_cluster_size=max_leiden_size,
    #     max_depth=5,
    # )

    # pickle leiden clusters for debugging
    import pickle
    from pathlib import Path
    leiden_path = Path(f"/home/lien002/nobackup_lien002/projects/clusterclue_publication/asdb5_motifs/output/temp/leiden_clusters_{leiden_resolution}_{min_leiden_size}_{max_leiden_size}.pkl")
    if not leiden_path.exists():
        logger.info(
            f"Starting recursive Leiden clustering "
            f"(min nodes {min_leiden_size}, max nodes {max_leiden_size})"
        )
        leiden_clusters = _recursive_leiden(
            G,
            resolution=leiden_resolution,
            min_cluster_size=min_leiden_size,
            max_cluster_size=max_leiden_size,
            max_depth=5,
        )
        with open(leiden_path, "wb") as f:
            pickle.dump(leiden_clusters, f)
        logger.info(f"Saved Leiden clusters to {leiden_path}")
    else:
        with open(leiden_path, "rb") as f:
            leiden_clusters = pickle.load(f)
            logger.info(f"Loaded precomputed Leiden clusters from {leiden_path}")

    n_clustered = sum(len(c) for c in leiden_clusters)
    logger.info(
        f"Recursive Leiden produced {len(leiden_clusters)} terminal clusters containing {n_clustered} ({n_clustered / G.number_of_nodes() * 100:.2f}%) nodes."
    )

    communities = aslpaw_within_leiden(
        G,
        leiden_clusters,
        min_comm_size=min_comm_size,
        log_queue=log_queue,
        cores=cores,
    )

    logger.info(
        f"ASLPAw found {len(communities)} overlapping communities "
        f"(min {min_comm_size} nodes each)"
    )
    # communities = leiden_lpam_clustering(
    #     G, 
    #     leiden_resolution=1.0, 
    #     min_leiden_size=20,
    #     max_leiden_edges=max_edges,
    #     min_comm_size=min_comm_size,
    # )
    # communities = leiden_aslpaw_clustering(
    #     G,
    #     leiden_resolution=1.0,
    #     min_leiden_size=20,
    #     max_leiden_edges=max_edges,
    #     min_comm_size=min_comm_size,
    # )
    # communities = leiden_clustering(
    #     G,
    #     leiden_resolution=2,
    #     min_leiden_size=20,
    #     max_leiden_size=100,
    #     min_comm_size=min_comm_size,
    # )

    # get the modules in each community
    communities_modules = [[modules[node] for node in comm] for comm in communities]
    n_in_communities = sum(len(c) for c in communities_modules)

    logger.info(
        f"Final output: {len(communities_modules)} overlapping communities "
        f"(min {min_comm_size} modules each), "
        f"total {n_in_communities} ({n_in_communities / len(modules) * 100:.2f}%) module assignments."
    )

    return communities_modules


def invert_communities(communities):
    """Convert community list into node → communities mapping"""
    membership = defaultdict(list)
    for cid, comm in enumerate(communities):
        for node in comm:
            membership[node].append(cid)
    return membership


# def _aslpaw_within_leiden(G, leiden_clusters, min_comm_size=5, log_queue=None, cores=1):
#     """
#     Run ASLPAw inside each Leiden cluster.
#     Returns a list of overlapping communities (global node IDs).
#     """
#     all_communities = []

#     for nodes in leiden_clusters:

#         subG = G.subgraph(nodes).copy()
#         E = subG.number_of_edges()

#         logger.info(
#             f"Running ASLPAw on Leiden cluster "
#             f"({len(nodes)} nodes, {E:,} edges)"
#         )

#         # ASLPAw directly on NetworkX graph
#         result = algorithms.aslpaw(subG)

#         sub_comms = [
#             list(comm)
#             for comm in result.communities
#             if len(comm) >= min_comm_size
#         ]

#         all_communities.extend(sub_comms)

#     return all_communities


def combined_worker_init(log_queue, G):
    """Call both your existing init + G sharing"""
    worker_init(log_queue)  # Your logging setup
    global _global_G
    _global_G = G


def process_cluster(nodes):
    """Worker function to run ASLPAw on a single Leiden cluster"""
    # log duration of this function
    logger.info(f"PID {os.getpid()} handling cluster with {len(nodes)} nodes")
    start_time = time.time()
    subG = _global_G.subgraph(nodes).copy()
    result = algorithms.aslpaw(subG)

    sub_comms = [
        list(comm)
        for comm in result.communities
    ]
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"ASLPAw on cluster with {len(nodes)} nodes took {duration:.2f} seconds. Returned {len(sub_comms)} communities.")

    return sub_comms


def aslpaw_within_leiden(G, leiden_clusters, min_comm_size=5, log_queue=None, cores=1):
    """
    Run ASLPAw inside each Leiden cluster.
    Returns a list of overlapping communities (global node IDs).
    """
    # limit to first 20 clusters for testing
    leiden_clusters = leiden_clusters[:20]
    cores = 10

    logger.info(f"Starting parallel ASLPAw on {len(leiden_clusters)} Leiden clusters using {cores} cores")

    pool = Pool(
        cores,
        maxtasksperchild=100,
        initializer=combined_worker_init,
        initargs=(log_queue, G)
    )
    with pool:
        results = pool.map(process_cluster, leiden_clusters)

    all_communities = [comm for comms in results for comm in comms if len(comm) > min_comm_size]
    return all_communities


# def leiden_clustering(
#     G,
#     leiden_resolution=1.0,
#     min_leiden_size=20,
#     max_leiden_size=100,
#     min_comm_size=5,
# ):
#     """Recursive Leiden  pipeline"""

#     # logger.info(
#     #     f"Starting recursive Leiden clustering "
#     #     f"(min nodes {min_leiden_size}, max edges {max_leiden_edges:,})"
#     # )

#     # leiden_clusters = recursive_leiden(
#     #     G,
#     #     resolution=leiden_resolution,
#     #     max_cluster_edges=max_leiden_edges,
#     #     min_cluster_size=min_leiden_size,
#     #     max_depth=5,
#     # )

#     # logger.info(
#     #     f"Recursive Leiden produced {len(leiden_clusters)} terminal clusters."
#     # )

#     # pickle leiden clusters for debugging
#     import pickle
#     from pathlib import Path
#     leiden_path = Path(f"/home/lien002/nobackup_lien002/projects/clusterclue_publication/asdb5_motifs/output/temp/leiden_clusters_{leiden_resolution}_{min_leiden_size}_{max_leiden_edges}.pkl")
#     if not leiden_path.exists():
#         logger.info(
#             f"Starting recursive Leiden clustering "
#             f"(min nodes {min_leiden_size}, max edges {max_leiden_edges:,})"
#         )

#         leiden_clusters = _recursive_leiden(
#             G,
#             resolution=leiden_resolution,
#             max_cluster_edges=max_leiden_edges,
#             min_cluster_size=min_leiden_size,
#             max_depth=5,
#         )

#         logger.info(
#             f"Recursive Leiden produced {len(leiden_clusters)} terminal clusters."
#         )
#         with open(leiden_path, "wb") as f:
#             pickle.dump(leiden_clusters, f)
#         logger.info(f"Saved Leiden clusters to {leiden_path}")
#     else:
#         with open(leiden_path, "rb") as f:
#             leiden_clusters = pickle.load(f)
#             logger.info(f"Loaded precomputed Leiden clusters from {leiden_path}")

#     return leiden_clusters
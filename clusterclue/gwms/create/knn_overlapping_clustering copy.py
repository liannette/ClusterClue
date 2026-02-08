"""
kNN graph–based overlapping clustering for sparse gene-module data.

Core ideas:
- Cosine similarity on sparse gene presence/absence
- Approximate kNN graph via FAISS (HNSW)
- Overlapping community detection via link communities

Designed for:
- Hundreds of thousands of modules
- High-dimensional sparse features
- Non-exclusive cluster membership
"""

import logging
from collections import defaultdict
import numpy as np
import networkx as nx
import faiss
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from cdlib import algorithms
from cdlib.algorithms import lpam

import igraph as ig
import leidenalg
import networkx as nx
from cdlib import NodeClustering


logger = logging.getLogger(__name__)


def create_sparse_matrix(modules):
    """Convert iterable of gene-sets into a sparse binary matrix.

    Args:
        modules: Each element is a list/set of genes in a module.

    Returns:
        A scipy.sparse.csr_matrix of shape (n_modules, n_genes).
    """
    mlb = MultiLabelBinarizer(sparse_output=True)
    X = mlb.fit_transform(modules)
    return X


def build_knn_graph(
    X,
    k=50,
    hnsw_m=32,
    faiss_threads=None,
):
    """Build a cosine kNN graph using FAISS (HNSW).

    Args:
        X: Module × feature matrix.
        k: Number of nearest neighbors per node.
        hnsw_m: HNSW graph connectivity (tradeoff speed/accuracy).

    Returns:
        A networkx.Graph weighted kNN graph.
    """
    n, d = X.shape

    logger.info("Normalizing feature matrix for cosine similarity")
    X = normalize(X, norm="l2", axis=1)

    if faiss_threads is not None:
        faiss.omp_set_num_threads(faiss_threads)

    logger.info(f"Building FAISS HNSW index (dims={d}, M={hnsw_m})")
    index = faiss.IndexHNSWFlat(d, hnsw_m)
    index.metric_type = faiss.METRIC_INNER_PRODUCT
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 128 

    # add vectors in batches to avoid memory issues
    batch_size = 50000
    logger.info("Adding vectors to FAISS index in batches")
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        block = np.ascontiguousarray(
            X[start:end].astype(np.float32).toarray()
        )
        index.add(block)

    # query graph in batches to avoid memory issues
    logger.info(f"Querying kNN graph (k={k})")
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        block = np.ascontiguousarray(
            X[start:end].astype(np.float32).toarray()
        )
        sims, nbrs = index.search(block, k + 1)

        logger.info(f"Block Max sim: {np.max(sims[:,1:]):.4f}, Mean sim: {np.mean(sims[:,1:]):.4f}")
        logger.info(f"Block Sample sims[0][1:5]: {sims[0][1:5]}")

        for i in range(end - start):
            src = start + i
            for dst, sim in zip(nbrs[i][1:], sims[i][1:]):
                if sim > 0:  # only keep positive similarities
                    G.add_edge(src, int(dst), weight=float(sim))

    logger.info(
        f"kNN graph built: {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges"
    )

    return G


# def find_overlapping_communities(
#     G,
#     min_comm_size=5,
# ):
#     """Detect overlapping communities.

#     Args:
#         G: kNN graph.
#         min_comm_size: Minimum community size to retain.

#     Returns:
#         A cdlib.NodeClustering with overlapping communities.
#     """
#     logger.info("Running link-community detection (overlapping)")
#     #communities = algorithms.hierarchical_link_community(G)
#     communities = lpam(G, overlapping=True)

#     # Filter very small communities
#     communities.communities = [
#         c for c in communities.communities
#         if len(c) >= min_comm_size
#     ]

#     logger.info(
#         f"Detected {len(communities.communities)} "
#         "overlapping communities"
#     )
#     return communities


def run_leiden(G, resolution=1.0):
    """
    Run Leiden clustering on a NetworkX graph.
    Returns node -> leiden_cluster mapping.
    """
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
    )

    node2leiden = {
        idx2node[i]: cid
        for i, cid in enumerate(partition.membership)
    }
    return node2leiden


def lpam_within_leiden(
    G,
    node2leiden,
    min_comm_size=5,
    min_leiden_size=20,
):
    """
    Run LPAM inside each Leiden cluster.
    
    Returns a list of overlapping communities (global node IDs).
    """
    leiden2nodes = {}
    for node, cid in node2leiden.items():
        leiden2nodes.setdefault(cid, []).append(node)

    all_communities = []
    skipped = 0

    for cid, nodes in leiden2nodes.items():
        if len(nodes) < min_leiden_size:
            skipped += 1
            continue

        subG = G.subgraph(nodes)

        try:
            comms = lpam(subG, overlapping=True)
        except Exception as e:
            logger.debug(f"LPAM failed for Leiden cluster {cid}: {e}")
            continue

        for comm in comms.communities:
            if len(comm) >= min_comm_size:
                # Map subgraph nodes back to global IDs
                all_communities.append(list(comm))

    return all_communities


def leiden_lpam_clustering(
    G,
    leiden_resolution=1.0,
    min_leiden_size=20,
    min_comm_size=5,
):
    """
    Full Leiden → LPAM nesting pipeline.
    """
    # Step 1: Leiden
    node2leiden = run_leiden(G, resolution=leiden_resolution)

    # Step 2: LPAM inside Leiden clusters
    communities = lpam_within_leiden(
        G,
        node2leiden,
        min_comm_size=min_comm_size,
        min_leiden_size=min_leiden_size,
    )

    return communities


def knn_overlapping_clustering(
    modules,
    knn_k=50,
    min_comm_size=5,
):
    """End-to-end pipeline for overlapping community detection.

    Converts modules → sparse matrix → kNN graph → overlapping communities.

    Args:
        modules: Gene modules.
        knn_k: Number of neighbors in kNN graph.
        min_comm_size: Minimum size of communities.
        similarity_threshold: Optional edge pruning threshold.

    Returns:
        list of lists
    """
    X = create_sparse_matrix(modules)
    logger.info(f"Sparse matrix shape: {X.shape}")

    G = build_knn_graph(
        X,
        k=knn_k,
    )
    communities = leiden_lpam_clustering(G)
    # communities = find_overlapping_communities(
    #     G,
    #     min_comm_size=min_comm_size,
    # )

    return communities


def invert_communities(communities):
    """Convert community list into node → communities mapping.

    Args:
        communities: A cdlib.NodeClustering object.

    Returns:
        A dict mapping node index to list of community IDs.
    """
    membership = defaultdict(list)
    for cid, comm in enumerate(communities):
        for node in comm:
            membership[node].append(cid)
    return membership

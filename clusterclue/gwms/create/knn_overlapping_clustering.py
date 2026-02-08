import logging
import numpy as np
import networkx as nx
from sklearn.preprocessing import MultiLabelBinarizer
from pathlib import Path
import pickle
import time
import os

from clusterclue.utils import worker_init
from multiprocessing import Pool
from functools import partial
from collections import defaultdict

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

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        block = X[start:end]

        # Compute intersection and union efficiently
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
        # if start % 50_000 == 0:
        #     logger.info(f"Processed {end}/{n} nodes")

    logger.info(f"Jaccard graph built (th={jaccard_threshold}): {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

# def split_large_component(X, component, max_size=1000, min_threshold=0.1, max_threshold=0.9, step=0.05):
#     """
#     Iteratively increase Jaccard threshold to split large connected component.
    
#     Args:
#         X: sparse matrix
#         component: list of node indices
#         max_size: maximum allowed component size
#         min_threshold: lowest threshold to try
#         max_threshold: highest threshold to try  
#         step: threshold increment step
    
#     Returns:
#         list of smaller components
#     """
#     if len(component) <= max_size:
#         return [component]
    
#     logger.info(f"Splitting large component (size={len(component)}) with iterative thresholds")
    
#     current_threshold = min_threshold
#     sub_components = [component]
    
#     while current_threshold <= max_threshold and any(len(c) > max_size for c in sub_components):
#         # Build subgraph with current threshold
#         G_sub = jaccard_cutoff_sparse(X[component], jaccard_threshold=current_threshold)
        
#         # Get new connected components
#         new_components = [list(c) for c in nx.connected_components(G_sub) 
#                          if len(c) >= 3]  # min size filter
        
#         logger.info(f"Threshold {current_threshold:.3f}: {len(sub_components)} → {len(new_components)} components")
        
#         if len(new_components) == 1 and len(new_components[0]) == len(component):
#             # No split achieved, increase threshold
#             current_threshold += step
#             continue
        
#         sub_components = new_components
#         current_threshold += step
    
#     # Final filter: keep only components above min size
#     final_components = [c for c in sub_components if len(c) > 5]
    
#     logger.info(f"Final split: {len(component)} → {len(final_components)} components "
#                f"(max size {max([len(c) for c in final_components]):.0f})")
    
#     return final_components

def split_large_component(X, component, max_size=1000, min_threshold=0.1, max_threshold=0.9, step=0.05):
    """
    RECURSIVELY split large connected component with iterative Jaccard thresholds.
    
    Args:
        X: sparse matrix (rows subset for component)
        component: list of node indices (global indices)
        max_size: maximum allowed component size
        min_threshold: lowest threshold to try
        max_threshold: highest threshold to try  
        step: threshold increment step
    
    Returns:
        list of smaller components (all <= max_size, global indices)
    """
    if len(component) <= max_size:
        return [component]
    
    current_threshold = min_threshold
    logger.info(f"Splitting large component (size={len(component)}) recursively with iterative thresholds")
    
    while current_threshold <= max_threshold:
        # Build subgraph ONLY for THIS component
        X_sub = X[component]
        G_sub = jaccard_cutoff_sparse(X_sub, jaccard_threshold=current_threshold)
        
        # Get connected components (local → global indices)
        local_components = [list(c) for c in nx.connected_components(G_sub) if len(c) >= 3]
        new_components = [[component[i] for i in local_comp] for local_comp in local_components]
        
        logger.info(f"  Threshold {current_threshold:.3f}: {len(component)} → {len(new_components)} components "
                   f"(max size: {max(len(c) for c in new_components) if new_components else 0})")
        
        max_new_size = max([len(c) for c in new_components]) if new_components else len(component)
        if max_new_size < len(component):
            # SUCCESS: Recursively split any remaining large components
            #logger.info(f"  Successful split at {current_threshold:.3f} → recursing on large subcomponents")
            all_small_components = []
            
            for sub_comp in new_components:
                # RECURSIVE CALL
                small_subs = split_large_component(
                    X, sub_comp, max_size, 
                    min_threshold=current_threshold + step,  # Start from this threshold
                    max_threshold=max_threshold, step=step
                )
                all_small_components.extend(small_subs)
            
            logger.info(f"  Recursive split complete: {len(component)} → {len(all_small_components)} final components")
            return [c for c in all_small_components if len(c) >= 5]
        
        current_threshold += step
    
    # Failed to split
    logger.warning(f"Failed to split {len(component)} nodes even at threshold {max_threshold}")
    return [component]


def jaccard_overlapping_clustering(modules, jaccard_threshold=0.7, min_comm_size=5, 
                                 max_comp_size=1000, threshold_step=0.02,
                                 log_queue=None, cores=1):
    """End-to-end pipeline with adaptive thresholding for large components"""
    
    jaccard_threshold = 0.6
    threshold_step = 0.05
    min_comm_size = 5
    max_comp_size = 2000

    min_size = min_comm_size

    modules_path = Path.cwd() / "modules.pkl"
    if modules_path.exists():
        with open(modules_path, "rb") as f:
            modules_picked = pickle.load(f)
        logger.info(f"Loaded modules from {modules_path}")
        # check if modules are the same
        if modules_picked != modules:
            logger.warning("Loaded modules differ from input modules!")
        else:
            logger.info("Loaded modules match input modules.")
    else:
        logger.info(f"Saving modules to {modules_path}")
        with open(modules_path, "wb") as f:
            pickle.dump(modules, f)

    X = create_sparse_matrix(modules)  # Keep X for splitting

    X_path = Path.cwd() / "sparse_matrix.pkl"
    if X_path.exists():
        with open(X_path, "rb") as f:
            X_loaded = pickle.load(f)
        logger.info(f"Loaded sparse matrix from {X_path}")
        # check if X_loaded is the same as X
        if (X != X_loaded).nnz != 0:
            logger.warning("Loaded sparse matrix differs from computed matrix!")
        else:
            logger.info("Loaded sparse matrix matches computed matrix.")
    else:
        logger.info(f"Saving sparse matrix to {X_path}")
        with open(X_path, "wb") as f:
            pickle.dump(X, f)


    # Load/create main graph
    #cwd = Path.cwd() 
    #graph_path = Path(f"/home/lien002/nobackup_lien002/projects/clusterclue_publication/asdb5_motifs/output/temp/jaccard_graph_{jaccard_threshold}.pkl")
    graph_path = Path.cwd() / f"jaccard_graph_{jaccard_threshold}.pkl"
    if not graph_path.exists():
        logger.info(f"Sparse matrix shape: {X.shape}")
        G = jaccard_cutoff_sparse(X, jaccard_threshold=jaccard_threshold)
        with open(graph_path, "wb") as f:
            pickle.dump(G, f)
        logger.info(f"Saved Jaccard graph to {graph_path}")
    else:
        with open(graph_path, "rb") as f:
            G = pickle.load(f)
            logger.info(f"Loaded precomputed Jaccard graph (threshold={jaccard_threshold}) from {graph_path}")

    logger.info(f"Initial graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    

    # Step 1: connected components with size-based splitting
    components = [list(c) for c in nx.connected_components(G) if len(c) >= min_size]
    logger.info(f"Initial: {len(components)} connected components >= {min_size} nodes")
    
    # Split large components iteratively
    all_small_components = []
    for comp in components:
        if len(comp) > max_comp_size:
            split_comps = split_large_component(
                X, comp, max_size=max_comp_size, 
                min_threshold=jaccard_threshold, 
                max_threshold=1.0, 
                step=threshold_step
            )
            all_small_components.extend(split_comps)
        else:
            all_small_components.append(comp)
    
    components = all_small_components
    sizes = [len(c) for c in components]
    logger.info(f"After splitting: {len(components)} components (min={min(sizes)}, "
               f"max={max(sizes)}, median={np.median(sizes):.0f})")

    # Step 2: parallel clique finding
    communities = parallel_clique_percolation(G, components, k=4, min_size=3, cores=cores, log_queue=log_queue)

    # Expand cores
    communities = expand_by_similarity(G, communities, min_connections=0.25, min_similarity=0.2)
    # map back to modules
    module_cliques = [[modules[i] for i in c] for c in communities]

    n_clustered_final = len(set(i for c in communities for i in c))
    logger.info(
        f"Final output: {len(module_cliques)} overlapping communities, "
        f"total {n_clustered_final} ({n_clustered_final / len(modules) * 100:.2f}%) nodes."
    )
    
    return module_cliques

def combined_worker_init(G, log_queue=None):
    if log_queue is not None:   
        worker_init(log_queue)
    global _global_G
    _global_G = G

def find_cliques_in_subgraph(component, min_size=5):
    """Find all cliques in the graph G with at least min_size nodes."""
    #logger.info(f"PID {os.getpid()}: Finding cliques in subgraph with {len(component)} nodes")
    start_time = time.time()

    G = _global_G.subgraph(component).copy()
    cliques = [c for c in nx.find_cliques(G) if len(c) >= min_size]

    elapsed = time.time() - start_time
    logger.info(f"PID {os.getpid()}: Found {len(cliques)} cliques (min {min_size} nodes each) in subgraph with {len(component)} nodes in {elapsed:.2f} seconds")
    return cliques

def parallel_clique_finding(G, components, min_size=5, cores=1, log_queue=None):
    """Find cliques in each component in parallel."""
    # Remove debug limits for production
    #components = components[:5]  # DEBUG: limit to first 5 components
    cores = min(cores, len(components))  # Don't use more cores than components

    logger.info(f"Starting parallel clique finding in {len(components)} components using {cores} cores")
    with Pool(
        processes=cores,
        initializer=combined_worker_init,
        initargs=(G, log_queue),
        maxtasksperchild=50,
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


def clique_percolation(component, k=3, min_size=3):
    """Find k-clique communities in the subgraph defined by component."""
    #logger.info(f"PID {os.getpid()}: Finding cliques (k={k}) in subgraph with {len(component)} nodes")
    start_time = time.time()

    G = _global_G.subgraph(component).copy()
    cliques = [c for c in nx.find_cliques(G)]

    #small_cliques = [c for c in cliques if len(c) < k]
    k_cliques = [c for c in cliques if len(c) == k]

    # Build clique graph (k-cliques as nodes, overlap≥k-1 as edges)
    clique_graph = nx.Graph()
    clique_graph.add_nodes_from(range(len(k_cliques)))
    
    overlap_threshold = k - 1
    for i, c1 in enumerate(k_cliques):
        for j, c2 in enumerate(k_cliques[i+1:], i+1):
            overlap = len(set(c1) & set(c2))
            if overlap >= overlap_threshold:
                clique_graph.add_edge(i, j)
    
    # Connected components = clique communities
    communities = []
    for comp in nx.connected_components(clique_graph):
        community_nodes = set()
        for i in comp:
            community_nodes.update(k_cliques[i])
        if len(community_nodes) >= min_size:
            communities.append(list(community_nodes))

    elapsed = time.time() - start_time
    #logger.info(f"PID {os.getpid()}: Found {len(communities)} communities (k={k}, min_size={min_size}) in subgraph of {len(component)} nodes in {elapsed:.2f} seconds")
    
    return communities


def parallel_clique_percolation(G, components, k=5, min_size=5, cores=1, log_queue=None):
    """Find cliques in each component in parallel."""
    # Remove debug limits for production
    #components = components[:5]  # DEBUG: limit to first 5 components
    cores = min(cores, len(components))  # Don't use more cores than components

    logger.info(f"Starting parallel clique percolation (k={k}, min_size={min_size}) in {len(components)} components using {cores} cores")
    with Pool(
        processes=cores,
        initializer=combined_worker_init,
        initargs=(G, None),
        maxtasksperchild=50,
    ) as pool:
        func = partial(clique_percolation, k=k, min_size=min_size)
        results = pool.map(func, components)

    # Flatten list of lists
    all_communities = [community for sublist in results for community in sublist]
    logger.info(f"Total communities found: {len(all_communities)}")
    return all_communities


def expand_by_similarity(G, communities, min_connections=0.3, min_similarity=0.25):
    """Add peripheral nodes strongly connected to communities"""
    expanded = []
    
    for comm in communities:
        core = set(comm)
        added_nodes = set()
        
        # Find candidates: high connectivity + similarity to core
        for node in G.nodes:
            if node in core:
                continue
                
            # Count connections to core
            core_neighbors = sum(1 for n in G.neighbors(node) if n in core)
            connection_ratio = core_neighbors / len(core)
            
            # Check average similarity to core neighbors
            core_nbr_weights = [G[node][n]['weight'] for n in G.neighbors(node) if n in core]
            if core_nbr_weights:
                avg_similarity = np.mean(core_nbr_weights)
                
                if (connection_ratio >= min_connections and 
                    avg_similarity >= min_similarity):
                    added_nodes.add(node)
        
        expanded.append(list(core | added_nodes))
    
    logger.info(f"Expanded {len(communities)} → {len(expanded)} communities, "
               f"added {sum(len(c)-len(comm) for c,comm in zip(expanded,communities))} nodes")
    return expanded

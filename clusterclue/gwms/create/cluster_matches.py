import logging
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
# import hdbscan
import umap
import matplotlib.pyplot as plt
from collections import Counter
# import faiss
#from scipy.sparse import csr_matrix


logger = logging.getLogger(__name__)


def create_sparse_matrix(modules):
    mlb = MultiLabelBinarizer(sparse_output=True)
    X = mlb.fit_transform(modules)
    return X


def reduce_dimensions_and_normalize(X, n_components=100, random_state=42):
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    X_reduced = svd.fit_transform(X).astype(np.float32)  # halves memory
    X_reduced = normalize(X_reduced, norm='l2')
    return X_reduced, svd.explained_variance_ratio_.cumsum()


def cluster_hdbscan(
    X, 
    min_cluster_size=5,
    cluster_selection_method='leaf',
    n_jobs=-1
    ):
    logger.info(
        f"Clustering with HDBSCAN (min_cluster_size={min_cluster_size}, "
        f"cluster_selection_method={cluster_selection_method}, n_jobs={n_jobs})"
        )
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
        metric='euclidean',
        cluster_selection_method=cluster_selection_method,
        core_dist_n_jobs=n_jobs,
        copy=False,
    )
    labels = clusterer.fit_predict(X)
    return labels


def cluster_hdbscan_sklearn(
    X,
    min_cluster_size=5,
    cluster_selection_method='eom',
    cluster_selection_epsilon=0.0,
    leaf_size=40,
    n_jobs=-1
    ):
    from sklearn.cluster import HDBSCAN as SklearnHDBSCAN

    logger.info(
        f"Clustering with sklearn HDBSCAN ("
        f"n_samples={X.shape[0]:,}, "
        f"n_features={X.shape[1]:,}, "
        f"min_cluster_size={min_cluster_size}, "
        f"cluster_selection_method={cluster_selection_method}, "
        f"cluster_selection_epsilon={cluster_selection_epsilon}, "
        f"leaf_size={leaf_size}, "
        f"n_jobs={n_jobs})"
    )
    clusterer = SklearnHDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
        metric='euclidean',
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=cluster_selection_epsilon,
        n_jobs=n_jobs,
        algorithm='ball_tree', # memory efficient tree structure
        leaf_size=leaf_size, # increase this to save memory
        copy=True,
    )
    labels = clusterer.fit_predict(X)

    return labels


# def cluster_hdbscan_precomputed(
#     X, 
#     min_cluster_size=5,
#     cluster_selection_method='leaf',
#     n_jobs=-1
#     ):
#     # This is an alternative approach that precomputes the nearest neighbor graph and passes it to HDBSCAN.
#     # It can be faster for large datasets, but requires more memory to store the distance matrix.
#     # uses multiple cores for nearest neighbor search, but HDBSCAN itself will run on the precomputed graph.
#     from sklearn.neighbors import NearestNeighbors

#     # This step is fully parallel
#     logger.info(
#         f"Precomputing nearest neighbors (n_neighbors={min_cluster_size}, n_jobs={n_jobs})"
#         )
#     nn = NearestNeighbors(
#         n_neighbors=min_cluster_size, 
#         metric='euclidean', 
#         n_jobs=n_jobs,      # actually uses all cores
#         algorithm='ball_tree'
#     )
#     nn.fit(X)
#     distances, indices = nn.kneighbors(X)

#     # Pass precomputed graph to HDBSCAN
#     clusterer = hdbscan.HDBSCAN(
#         min_cluster_size=min_cluster_size,
#         min_samples=1,
#         metric='precomputed',    # tell HDBSCAN distances are already computed
#         cluster_selection_method=cluster_selection_method,
#         core_dist_n_jobs=n_jobs
#     )
#     # Build sparse distance matrix from nn results
#     from sklearn.neighbors import kneighbors_graph
#     graph = kneighbors_graph(X, n_neighbors=min_cluster_size, 
#                             metric='euclidean', n_jobs=n_jobs)
#     logger.info(
#         f"Clustering with HDBSCAN (min_cluster_size={min_cluster_size}, "
#         f"cluster_selection_method={cluster_selection_method}, n_jobs={n_jobs})"
#         )
#     labels = clusterer.fit_predict(graph.toarray())
#     return labels


# def precompute_nn_faiss(X, n_neighbors=5, n_jobs=64):
#     """Compute approximate nearest neighbors using FAISS."""
#     if hasattr(X, 'toarray'):
#         X_f32 = np.ascontiguousarray(X.toarray(), dtype=np.float32)
#     else:
#         X_f32 = np.ascontiguousarray(X, dtype=np.float32)

#     n_samples, n_dims = X_f32.shape
#     print(f"Building FAISS index (n={n_samples}, d={n_dims})")

#     n_voronoi = min(int(np.sqrt(n_samples)), 4096)
#     quantizer = faiss.IndexFlatL2(n_dims)
#     index     = faiss.IndexIVFFlat(quantizer, n_dims, n_voronoi, faiss.METRIC_L2)
#     faiss.omp_set_num_threads(n_jobs)

#     print(f"Training FAISS index with {n_voronoi} Voronoi cells...")
#     index.train(X_f32)
#     index.add(X_f32)
#     index.nprobe = min(64, n_voronoi)

#     print(f"Searching {n_neighbors} nearest neighbors...")
#     distances, indices = index.search(X_f32, n_neighbors + 1)  # +1 for self

#     # Remove self-matches
#     distances = distances[:, 1:]
#     indices   = indices[:, 1:]

#     return distances, indices, index, X_f32  # return index and X_f32 for retry


# def build_sparse_distance_matrix(distances, indices, n_samples):
#     """Convert NN results to symmetric sparse distance matrix."""
#     rows = np.repeat(np.arange(n_samples), distances.shape[1])
#     cols = indices.flatten()
#     data = np.sqrt(np.maximum(distances.flatten(), 0))  # sqrt of squared L2, clip negatives

#     matrix = csr_matrix((data, (rows, cols)), shape=(n_samples, n_samples))
#     matrix = matrix.maximum(matrix.T)  # symmetrize
#     return matrix


# def ensure_connected(dist_matrix, X_f32, index, initial_n_neighbors, n_jobs, max_neighbors=50):
#     """
#     Iteratively increase n_neighbors until the graph is fully connected.
#     Reuses the already-built FAISS index — no retraining needed.
#     """
#     from scipy.sparse.csgraph import connected_components

#     n_samples = dist_matrix.shape[0]
#     n_neighbors = initial_n_neighbors
#     dist_matrix_current = dist_matrix

#     while n_neighbors <= max_neighbors:
#         n_components, component_labels = connected_components(
#             dist_matrix_current, directed=False
#         )
#         component_sizes = np.bincount(component_labels)
#         print(
#             f"n_neighbors={n_neighbors}: {n_components} components "
#             f"(largest={component_sizes.max()}, smallest={component_sizes.min()})"
#         )

#         if n_components == 1:
#             print("Graph is fully connected")
#             return dist_matrix_current

#         # Increase n_neighbors and re-search — reuses trained index, fast
#         n_neighbors += 5
#         print(f"Graph not connected — retrying with n_neighbors={n_neighbors}...")
#         distances, indices = index.search(X_f32, n_neighbors + 1)
#         distances = distances[:, 1:]
#         indices   = indices[:, 1:]
#         dist_matrix_current = build_sparse_distance_matrix(distances, indices, n_samples)

#     # Last resort — connect remaining isolated points directly
#     print(
#         f"Could not fully connect graph with n_neighbors={max_neighbors}. "
#         f"Connecting remaining isolated points directly..."
#     )
#     return connect_isolated_points(dist_matrix_current, X_f32, component_labels, n_components)


# def connect_isolated_points(dist_matrix, X_f32, component_labels, n_components):
#     """
#     For any remaining isolated components, add a single edge to the
#     nearest point in the largest component. O(isolated_points) not O(n^2).
#     """
#     import faiss
#     from scipy.sparse import lil_matrix

#     component_sizes = np.bincount(component_labels)
#     main_component  = np.argmax(component_sizes)
#     main_mask       = component_labels == main_component
#     main_indices    = np.where(main_mask)[0]

#     # Build a small flat index over just the main component — fast exact search
#     X_main  = X_f32[main_mask]
#     index_main = faiss.IndexFlatL2(X_f32.shape[1])
#     index_main.add(X_main)

#     dist_lil = lil_matrix(dist_matrix)
#     n_connected = 0

#     for comp_id in range(n_components):
#         if comp_id == main_component:
#             continue

#         comp_mask    = component_labels == comp_id
#         comp_indices = np.where(comp_mask)[0]
#         X_comp       = X_f32[comp_mask]

#         # Find nearest neighbor in main component
#         dists, nn_in_main = index_main.search(X_comp, 1)
#         dists     = np.sqrt(dists.flatten())
#         nn_in_main = nn_in_main.flatten()

#         # Connect each isolated point to its nearest main component neighbor
#         for local_i, (d, main_local_i) in enumerate(zip(dists, nn_in_main)):
#             i = comp_indices[local_i]
#             j = main_indices[main_local_i]
#             dist_lil[i, j] = d
#             dist_lil[j, i] = d
#             n_connected += 1

#     print(f"Added {n_connected} edges to connect isolated points")
#     return dist_lil.tocsr()


# def cluster_hdbscan_faiss(
#     X,
#     min_cluster_size=5,
#     cluster_selection_method='leaf',
#     n_jobs=-1
# ):
#     print(
#         f"Clustering with HDBSCAN (min_cluster_size={min_cluster_size}, "
#         f"cluster_selection_method={cluster_selection_method}, n_jobs={n_jobs})"
#     )

#     n_samples       = X.shape[0]
#     initial_n_neighbors = max(min_cluster_size, int(np.log(n_samples)) + 1)

#     # Step 1: Build FAISS index and initial NN graph
#     print(f"Precomputing approximate nearest neighbors with FAISS "
#                 f"(n_neighbors={initial_n_neighbors})...")
#     distances, indices, faiss_index, X_f32 = precompute_nn_faiss(
#         X, n_neighbors=initial_n_neighbors, n_jobs=n_jobs
#     )
#     dist_matrix = build_sparse_distance_matrix(distances, indices, n_samples)

#     # Free NN results immediately after building distance matrix
#     del distances, indices
#     gc.collect()

#     # Step 2: Ensure connectivity — reuses FAISS index, no retraining
#     dist_matrix = ensure_connected(
#         dist_matrix, X_f32, faiss_index,
#         initial_n_neighbors=initial_n_neighbors,
#         n_jobs=n_jobs,
#         max_neighbors=50
#     )

#     # Free FAISS structures after connectivity is ensured
#     del X_f32, faiss_index
#     gc.collect()

#     # # Step 3: Cluster
#     # clusterer = hdbscan.HDBSCAN(
#     #     min_cluster_size=min_cluster_size,
#     #     min_samples=1,
#     #     metric='precomputed',
#     #     cluster_selection_method=cluster_selection_method,
#     #     core_dist_n_jobs=n_jobs,
#     #     approx_min_span_tree=True,
#     #     gen_min_span_tree=True
#     # )
#     # labels = clusterer.fit_predict(dist_matrix)
#     # return labels
#     clusterer = SklearnHDBSCAN(
#         min_cluster_size=min_cluster_size,
#         min_samples=1,
#         metric='precomputed',
#         cluster_selection_method=cluster_selection_method,
#         n_jobs=n_jobs,
#         store_centers=None,   # don't store centers — saves RAM
#         # gen_min_span_tree=True  # only enable if you actually need MST plots
#     )
#     labels = clusterer.fit_predict(dist_matrix)

#     del dist_matrix
#     gc.collect()


def generate_cluster_report(label_set, method, save_path=None):
    """
    Generates a single-method cluster size analysis report with 3 panels:
    - Cluster size histogram
    - Ranked cluster sizes (log scale)
    - Summary statistics

    """
    # --- Separate noise from real clusters ---
    is_noise    = label_set == -1
    clean_labels = label_set[~is_noise]
    counts      = Counter(clean_labels)
    sizes       = np.array(sorted(counts.values(), reverse=True))
    n_clusters  = len(counts)
    n_noise     = is_noise.sum()
    noise_pct   = n_noise / len(label_set) * 100
    percentiles = np.percentile(sizes, [25, 50, 75, 90, 99])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Cluster Size Report — {method}', fontsize=14, fontweight='bold')

    # --- Panel 1: Histogram ---
    ax = axes[0]
    ax.hist(sizes, bins=50, color='steelblue', edgecolor='white', linewidth=0.5)
    ax.axvline(np.median(sizes), color='red',    linestyle='--', linewidth=1.5,
               label=f'Median: {np.median(sizes):.0f}')
    ax.axvline(np.mean(sizes),   color='orange', linestyle='--', linewidth=1.5,
               label=f'Mean: {np.mean(sizes):.0f}')
    ax.set_title('Cluster size distribution')
    ax.set_xlabel('Cluster size (n points)')
    ax.set_ylabel('Number of clusters')
    ax.legend(fontsize=8)

    # --- Panel 2: Rank plot ---
    ax = axes[1]
    ax.plot(range(len(sizes)), sizes, color='steelblue', linewidth=1)
    ax.fill_between(range(len(sizes)), sizes, alpha=0.3, color='steelblue')
    ax.set_title('Ranked cluster sizes')
    ax.set_xlabel('Cluster rank (largest → smallest)')
    ax.set_ylabel('Cluster size (n points)')
    ax.set_yscale('log')

    # --- Panel 3: Summary stats ---
    ax = axes[2]
    ax.axis('off')
    stats_text = (
        f"Method: {method}\n"
        f"{'─' * 28}\n"
        f"Total points:     {len(label_set):>10,}\n"
        f"Clusters:         {n_clusters:>10,}\n"
        f"Noise points:     {n_noise:>10,} ({noise_pct:.1f}%)\n"
        f"\nCluster sizes:\n"
        f"  Min:            {sizes.min():>10,}\n"
        f"  Max:            {sizes.max():>10,}\n"
        f"  Mean:           {sizes.mean():>10.1f}\n"
        f"  Median:         {np.median(sizes):>10.1f}\n"
        f"  Std:            {sizes.std():>10.1f}\n"
        f"\nPercentiles:\n"
        f"  25th:           {percentiles[0]:>10.0f}\n"
        f"  50th:           {percentiles[1]:>10.0f}\n"
        f"  75th:           {percentiles[2]:>10.0f}\n"
        f"  90th:           {percentiles[3]:>10.0f}\n"
        f"  99th:           {percentiles[4]:>10.0f}\n"
        f"\nSize buckets:\n"
        f"  1-10:           {(sizes <= 10).sum():>10,}\n"
        f"  11-50:          {((sizes > 10)  & (sizes <= 50)).sum():>10,}\n"
        f"  51-200:         {((sizes > 50)  & (sizes <= 200)).sum():>10,}\n"
        f"  201-1000:       {((sizes > 200) & (sizes <= 1000)).sum():>10,}\n"
        f"  1000+:          {(sizes > 1000).sum():>10,}\n"
    )
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved cluster report to {save_path}")


def plot_umap_clusters(X_svd, labels, method, save_path=None):

    reducer = umap.UMAP(
        n_components=2, 
        random_state=42, 
        n_neighbors=30, 
        min_dist=0.1,
        )
    
    # subset for faster plotting
    n_subsample = min(50_000, len(X_svd))
    idx = np.random.choice(len(X_svd), size=n_subsample, replace=False)
    X_2d = reducer.fit_transform(X_svd[idx])
    labels_sub = labels[idx]

    fig, ax = plt.subplots(figsize=(10, 8))

    noise_mask = labels_sub == -1
    cluster_mask = ~noise_mask

    # Plot noise points without cluster coloring
    ax.scatter(
        X_2d[noise_mask, 0], X_2d[noise_mask, 1],
        c='lightgrey',
        s=2,
        alpha=0.2,
        linewidths=0
    )

    # Plot only non-noise points with cluster colors
    scatter = ax.scatter(
        X_2d[cluster_mask, 0], X_2d[cluster_mask, 1],
        c=labels_sub[cluster_mask],
        cmap='tab20',
        s=2,
        alpha=0.5,
        linewidths=0
    )

    plt.colorbar(scatter, ax=ax, label='Cluster label')
    ax.set_title(f'{method} — UMAP projection (subsampled {len(X_svd)} -> {n_subsample})')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"UMAP projection saved to {save_path}")

    
def cluster_modules(modules, n_components=100, outdir=None, n_cores=-1):
    """Cluster modules using HDBSCAN and generate visualization reports.
    
    Converts a list of modules into a sparse matrix, reduces dimensions using SVD,
    applies HDBSCAN clustering, and optionally generates cluster reports and UMAP
    visualizations.
    
    Args:
        modules: List of iterables containing module assignments for each sample.
        n_components (int, optional): Number of components to keep after dimensionality reduction. Defaults to 100.
        outdir (str, optional): Output directory for saving cluster report and UMAP
            visualizations. If None, visualizations are not saved. Defaults to None.
    
    Returns:
        numpy.ndarray: Cluster labels for each sample. -1 indicates noise points
            not assigned to any cluster by HDBSCAN.
    
    Raises:
        None
    """
    logger.info(f"Clustering the subcluster modules into motifs...")

    X = create_sparse_matrix(modules)
    logger.info(f"Created sparse matrix with shape {X.shape} and {X.nnz} non-zero entries")

    n_components = min(n_components, X.shape[1]-1) # can't have more components than features
    X_reduced, cumvar = reduce_dimensions_and_normalize(X, n_components)
    logger.info(f"Reduced dimensionality to {n_components} components, preserving {cumvar[-1]*100:.1f}% of variance")

    method = 'eom'
    epsilon = 0.1
    labels = cluster_hdbscan_sklearn(
        X=X_reduced, 
        min_cluster_size=5,
        cluster_selection_method=method,
        cluster_selection_epsilon=epsilon,
        n_jobs=n_cores
        )
    # -1 labels = noise points HDBSCAN didn't assign to any cluster
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    pct_noise = n_noise / len(labels) * 100
    logger.info(f"Found {n_clusters} clusters, {n_noise} noise points ({pct_noise:.1f}%)")

    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)
        generate_cluster_report(
            labels, 
            f"HDBSCAN ({method}, epsilon={epsilon})",
            save_path=Path(outdir) /f'clusters_report_{method}_comp{n_components}.png'
            )
        plot_umap_clusters(
            X_reduced, 
            labels, 
            f"HDBSCAN ({method}, epsilon={epsilon})", 
            save_path=Path(outdir) /f'clusters_umap_{method}_comp{n_components}.png',
        )

    return labels
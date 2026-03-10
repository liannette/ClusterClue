import logging
import gc
import time
import numpy as np
import matplotlib.pyplot as plt
import umap
import psutil
import os
from collections import Counter
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize, MultiLabelBinarizer
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

logger = logging.getLogger(__name__)

NUMEXPR_MAX_THREADS = min(16, os.cpu_count())  # to suppress numexpr warnings

# ─────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────

def log_memory(label=""):
    process = psutil.Process(os.getpid())
    ram_gb  = process.memory_info().rss / 1024**3
    logger.info(f"[RAM] {label}: {ram_gb:.2f} GB")
    return ram_gb


def log_matrix_stats(X, label="Matrix"):
    if hasattr(X, 'nnz'):
        density = X.nnz / (X.shape[0] * X.shape[1]) * 100
        mem_mb  = (X.data.nbytes + X.indices.nbytes + X.indptr.nbytes) / 1024**2
        avg_nnz = X.nnz / X.shape[0]
        logger.info(
            f"[MATRIX] {label}: shape={X.shape}, nnz={X.nnz:,}, "
            f"density={density:.4f}%, avg_nnz_per_row={avg_nnz:.1f}, "
            f"memory={mem_mb:.1f} MB"
        )
    else:
        mem_mb = X.nbytes / 1024**2
        logger.info(
            f"[MATRIX] {label}: shape={X.shape}, dtype={X.dtype}, "
            f"memory={mem_mb:.1f} MB"
        )


# ─────────────────────────────────────────────────────────────
# 0. SPARSE MATRIX CREATION
# ─────────────────────────────────────────────────────────────

def create_sparse_matrix(modules):
    """Convert list of module token sets to sparse binary matrix."""
    logger.info(f"Creating sparse matrix from {len(modules):,} modules...")
    log_memory("before sparse matrix")
    t_start = time.time()

    mlb = MultiLabelBinarizer(sparse_output=True)
    X   = mlb.fit_transform(modules)
    elapsed = time.time() - t_start

    log_matrix_stats(X, "sparse binary matrix")
    log_memory("after sparse matrix")
    logger.info(f"Created sparse matrix in {elapsed:.1f}s — {len(mlb.classes_):,} unique tokens")

    nnz_per_row = np.diff(X.indptr)
    pcts        = np.percentile(nnz_per_row, [1, 5, 25, 50, 75, 95, 99])
    logger.info(
        f"Tokens per module: "
        f"min={nnz_per_row.min()}, max={nnz_per_row.max()}, "
        f"mean={nnz_per_row.mean():.1f}, median={np.median(nnz_per_row):.1f}\n"
        f"  percentiles: 1pct={pcts[0]:.0f}, 5pct={pcts[1]:.0f}, "
        f"25pct={pcts[2]:.0f}, 50pct={pcts[3]:.0f}, 75pct={pcts[4]:.0f}, "
        f"95pct={pcts[5]:.0f}, 99pct={pcts[6]:.0f}"
    )
    return X, mlb


# ─────────────────────────────────────────────────────────────
# 1. OPTIONAL DIMENSIONALITY REDUCTION
# ─────────────────────────────────────────────────────────────

def reduce_to_variance_threshold(
    X, target_variance=0.50, step=50, max_components=3000, random_state=42
):
    """Finds minimum n_components explaining target_variance.
    Output is L2-normalized — euclidean distance == cosine distance.
    """
    max_components = min(max_components, X.shape[1] - 1)
    step           = min(step, max_components)
    best_n         = None
    cumvar         = 0.0
    n              = step

    logger.info(
        f"Searching for n_components explaining {target_variance*100:.0f}% variance "
        f"(step={step}, max={max_components})..."
    )
    log_memory("before SVD")
    t_start = time.time()

    while n <= max_components:
        svd    = TruncatedSVD(n_components=n, random_state=random_state)
        svd.fit(X)
        cumvar = svd.explained_variance_ratio_.sum()
        logger.info(f"  n_components={n:>5}: {cumvar*100:.2f}% variance explained")

        if cumvar >= target_variance:
            cumvar_per_component = svd.explained_variance_ratio_.cumsum()
            best_n   = int((cumvar_per_component >= target_variance).argmax()) + 1
            achieved = cumvar_per_component[best_n - 1] * 100
            logger.info(
                f"Target reached — refined to {best_n} components ({achieved:.2f}% variance)"
            )
            break
        n += step

    if best_n is None:
        best_n = min(n - step, max_components)
        logger.warning(
            f"Could not reach {target_variance*100:.0f}% variance within "
            f"{max_components} components. Best: {cumvar*100:.2f}% at "
            f"n_components={best_n}."
        )

    logger.info(f"Fitting final SVD with n_components={best_n}...")
    svd       = TruncatedSVD(n_components=best_n, random_state=random_state)
    X_reduced = svd.fit_transform(X).astype(np.float32)
    X_reduced = normalize(X_reduced, norm='l2')
    elapsed   = time.time() - t_start

    log_matrix_stats(X_reduced, f"SVD output (n_components={best_n}, L2-normalized)")
    log_memory("after SVD")
    logger.info(f"Dimensionality reduction completed in {elapsed:.1f}s")

    return X_reduced, best_n, svd.explained_variance_ratio_.cumsum()


# ─────────────────────────────────────────────────────────────
# 2. BATCH SIZE
# ─────────────────────────────────────────────────────────────

def get_auto_batch_size(n_samples):
    """Automatically determine batch size for MiniBatchKMeans.
    Uses ~5% of n_samples, capped between 500 and 50000.
    """
    batch = max(500, int(n_samples * 0.05))
    batch = min(batch, 50_000)
    logger.info(
        f"Auto batch size: {batch:,} ({batch/n_samples*100:.1f}% of {n_samples:,} samples)"
    )
    return batch


# ─────────────────────────────────────────────────────────────
# 3. CLUSTERING
# ─────────────────────────────────────────────────────────────

def minibatch_kmeans_clustering(X, k, batch_size=None, n_init=3, random_state=42):
    """Cluster X using MiniBatchKMeans.

    Args:
        X: Array-like of shape (n_samples, n_features). Dense or sparse.
            SVD-reduced dense input is recommended over raw sparse.
        k: Number of clusters.
        batch_size: Mini-batch size. Auto-determined if None.
        n_init: Number of initializations. Best result is kept.
            Defaults to 3.
        random_state: Random seed for reproducibility. Defaults to 42.

    Returns:
        Tuple of (labels, kmeans) where labels is an np.ndarray of shape
        (n_samples,) with cluster assignments in range [0, k-1], and
        kmeans is the fitted MiniBatchKMeans object.
    """
    if batch_size is None:
        batch_size = get_auto_batch_size(X.shape[0])

    logger.info(
        f"Clustering {X.shape[0]:,} modules with k={k:,} "
        f"(batch_size={batch_size:,}, n_init={n_init})"
    )
    log_memory(f"before KMeans k={k}")

    kmeans = MiniBatchKMeans(
        n_clusters=int(k),
        batch_size=batch_size,
        init='k-means++',
        n_init=n_init,
        max_iter=1000,
        tol=1e-4,
        random_state=random_state,
    )

    t_start = time.time()
    kmeans.fit(X)
    elapsed = time.time() - t_start

    log_memory(f"after KMeans k={k}")
    logger.info(
        f"KMeans converged in {kmeans.n_iter_} iterations "
        f"(inertia: {kmeans.inertia_:.2f}, time: {elapsed:.1f}s)"
    )

    if kmeans.n_iter_ == 1:
        logger.warning(
            f"KMeans converged in 1 iteration for k={k} — this may indicate "
            f"the data representation is too sparse/compressed. "
            f"Consider enabling SVD reduction."
        )

    # Cluster size stats
    sizes   = np.array(sorted(Counter(kmeans.labels_).values(), reverse=True))
    pcts    = np.percentile(sizes, [25, 50, 75, 90, 99])
    buckets = {
        '1-10':     (sizes <= 10).sum(),
        '11-50':    ((sizes > 10)  & (sizes <= 50)).sum(),
        '51-200':   ((sizes > 50)  & (sizes <= 200)).sum(),
        '201-1000': ((sizes > 200) & (sizes <= 1000)).sum(),
        '1000+':    (sizes > 1000).sum(),
    }
    logger.info(
        f"Found {k:,} clusters for {X.shape[0]:,} modules\n"
        f"  size: min={sizes.min()}, max={sizes.max()}, "
        f"mean={sizes.mean():.1f}, median={np.median(sizes):.1f}, "
        f"std={sizes.std():.1f}\n"
        f"  percentiles: 25th={pcts[0]:.0f}, 50th={pcts[1]:.0f}, "
        f"75th={pcts[2]:.0f}, 90th={pcts[3]:.0f}, 99th={pcts[4]:.0f}\n"
        f"  buckets: {buckets}"
    )

    return kmeans.labels_, kmeans


# ─────────────────────────────────────────────────────────────
# 4. EVALUATION
# ─────────────────────────────────────────────────────────────

def evaluate_kmeans(X, labels, method_name, sample_size=10_000):
    """Evaluate KMeans clustering quality.

    Args:
        X: Array of shape (n_samples, n_features) used for clustering.
        labels: Cluster labels of shape (n_samples,).
        method_name: Label for logging.
        sample_size: Max points to use for silhouette score. Defaults to 10000.

    Returns:
        Dict with keys 'method', 'k', 'silhouette', 'calinski_harabasz',
        'davies_bouldin', or None if fewer than 2 clusters.
    """
    n_clusters = len(set(labels))

    if n_clusters < 2:
        logger.warning(f"Cannot evaluate {method_name} — fewer than 2 clusters")
        return None

    logger.info(f"Evaluating {method_name}...")
    sample_size = min(sample_size, len(labels))

    t_start = time.time()
    sil = silhouette_score(X, labels, metric='euclidean', sample_size=sample_size)
    db  = davies_bouldin_score(X, labels)
    ch  = calinski_harabasz_score(X, labels)
    elapsed = time.time() - t_start

    logger.info(
        f"k={n_clusters}: silhouette={sil:.4f}, "
        f"calinski_harabasz={ch:.4f}, davies_bouldin={db:.4f} "
        f"({elapsed:.1f}s)\n"
        f"  silhouette: "
        f"{'good (>0.2)' if sil > 0.2 else 'moderate (0-0.2)' if sil > 0 else 'poor (<0)'}, "
        f"davies_bouldin: "
        f"{'good (<1.0)' if db < 1.0 else 'moderate (1-2)' if db < 2.0 else 'poor (>2)'}"
    )

    return {
        'method': method_name, 'k': n_clusters,
        'silhouette': sil, 'calinski_harabasz': ch, 'davies_bouldin': db,
    }


# ─────────────────────────────────────────────────────────────
# 5. VISUALIZATION
# ─────────────────────────────────────────────────────────────

def plot_cluster_report(labels, method, save_path):
    """Cluster size report: histogram + rank plot + summary stats."""
    counts      = Counter(labels)
    sizes       = np.array(sorted(counts.values(), reverse=True))
    n_clusters  = len(counts)
    percentiles = np.percentile(sizes, [25, 50, 75, 90, 99])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Cluster Size Report — {method}', fontsize=14, fontweight='bold')

    ax = axes[0]
    ax.hist(sizes, bins=50, color='steelblue', edgecolor='white', linewidth=0.5)
    ax.axvline(np.median(sizes), color='red', linestyle='--', linewidth=1.5,
               label=f'Median: {np.median(sizes):.0f}')
    ax.axvline(np.mean(sizes), color='orange', linestyle='--', linewidth=1.5,
               label=f'Mean: {np.mean(sizes):.0f}')
    ax.set_title('Cluster size distribution')
    ax.set_xlabel('Cluster size (n points)')
    ax.set_ylabel('Number of clusters')
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(range(len(sizes)), sizes, color='steelblue', linewidth=1)
    ax.fill_between(range(len(sizes)), sizes, alpha=0.3, color='steelblue')
    ax.set_title('Ranked cluster sizes')
    ax.set_xlabel('Cluster rank (largest → smallest)')
    ax.set_ylabel('Cluster size (n points)')
    ax.set_yscale('log')

    ax = axes[2]
    ax.axis('off')
    stats_text = (
        f"Method: {method}\n"
        f"{'─' * 28}\n"
        f"Total points:     {len(labels):>10,}\n"
        f"Clusters:         {n_clusters:>10,}\n"
        f"Noise points:     {'N/A (KMeans)':>10}\n"
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
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved cluster report to {save_path}")


def compute_umap_embedding(X, n_subsample=50_000, n_jobs=1, random_state=42):
    """Compute 2D UMAP embedding once for reuse across multiple label sets.

    Args:
        X: Dense array of shape (n_samples, n_features). Should be
            L2-normalized SVD output for consistency with clustering metric.
        n_subsample: Maximum number of points to use for UMAP. If X has
            fewer points, all are used. Defaults to 50000.
        n_jobs: Number of parallel jobs for UMAP. Note that setting n_jobs > 1
            requires random_state=None, sacrificing reproducibility.
            Defaults to 1.
        random_state: Random seed for reproducibility. Ignored if n_jobs > 1.
            Defaults to 42.

    Returns:
        A tuple of (X_2d, idx) where:
        X_2d: 2D UMAP embedding of shape (n_subsample, 2).
        idx: Indices of the subsampled points in the original X, shape
            (n_subsample,). Use these to index into label arrays when plotting.
    """
    n_subsample = min(n_subsample, len(X))
    idx         = np.random.default_rng(random_state).choice(
                      len(X), size=n_subsample, replace=False
                  )
    X_sub = X[idx]

    logger.info(f"Computing UMAP embedding on {n_subsample:,} points (metric=euclidean)...")
    log_memory("before UMAP")
    t_start = time.time()

    reducer = umap.UMAP(
        n_components=2,
        random_state=random_state if n_jobs == 1 else None,
        n_neighbors=30,
        min_dist=0.1,
        n_jobs=n_jobs,
        metric='euclidean',
    )
    X_2d    = reducer.fit_transform(X_sub)
    elapsed = time.time() - t_start

    logger.info(f"UMAP embedding completed in {elapsed:.1f}s")
    log_memory("after UMAP")

    return X_2d, idx


def plot_umap_clusters(X_2d, idx, labels, method, save_path):
    """Plot a precomputed UMAP embedding colored by cluster labels.

    Uses a precomputed embedding from compute_umap_embedding rather than
    recomputing UMAP for each label set. This avoids redundant computation
    and memory accumulation when comparing multiple k values.

    Args:
        X_2d: Precomputed 2D UMAP embedding of shape (n_subsample, 2),
            as returned by compute_umap_embedding.
        idx: Subsample indices of shape (n_subsample,) as returned by
            compute_umap_embedding. Used to extract the correct labels
            for the subsampled points.
        labels: Full cluster label array of shape (n_samples,). Will be
            indexed by idx to get labels for the subsampled points.
        method: Method name used in the plot title.
        save_path: Path to save the figure. If None, figure is shown
            but not saved. Defaults to None.
    """
    labels_sub  = labels[idx]
    n_subsample = len(idx)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=labels_sub, cmap='tab20',
        s=2, alpha=0.5, linewidths=0
    )
    plt.colorbar(scatter, ax=ax, label='Cluster label')
    ax.set_title(f'{method} — UMAP projection (n={n_subsample:,})')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"UMAP projection saved to {save_path}")


# ─────────────────────────────────────────────────────────────
# 6. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def run_kmeans_pipeline(
    modules,
    k_values,
    use_svd          = True,
    target_variance  = 0.50,
    n_init           = 3,
    n_jobs           = 1,
    outdir           = None,
    subsample_n      = None,
):
    """Full KMeans clustering pipeline from raw module token lists to cluster labels.

    Args:
        modules: List of iterables of token strings. Each element is a collection
            of token strings for one module.
            e.g. [['geneA', 'geneB'], ['geneB', 'geneC'], ...]
        k_values: List of k values to evaluate. Each k is run as a separate
            MiniBatchKMeans clustering. e.g. [500, 1000, 1500, 2000]
        use_svd: If True, applies TruncatedSVD before clustering. If False,
            clusters directly on the sparse binary matrix. Warning: clustering
            on raw sparse data likely causes 1-iteration convergence due to
            distance collapse. Defaults to True.
        target_variance: Cumulative explained variance threshold for SVD. The
            minimum number of components explaining this fraction of variance
            is selected automatically. Only used if use_svd=True.
            Defaults to 0.50.
        n_init: Number of KMeans initializations per k. The best result across
            initializations is kept. Higher values improve stability but increase
            runtime. Defaults to 3.
        n_jobs: Parallelism for UMAP visualization. Note that MiniBatchKMeans
            does not benefit significantly from parallelism and always runs
            single-threaded. Defaults to 1.
        outdir: Directory to save cluster size reports and UMAP plots for each
            k value. Directory is created if it does not exist. If None, plots
            are shown but not saved. Defaults to None.
        subsample_n: If set, subsamples this many points from the sparse matrix
            before SVD and clustering. Useful for fast development iteration.
            Set to None for production runs on the full dataset. Defaults to None.

    Returns:
        A tuple of (all_labels, mlb, results) where:

        all_labels: Dict mapping k to cluster label arrays {k: np.ndarray}.
            Labels are integers in range [0, k-1]. All points are assigned —
            KMeans has no noise points. e.g. all_labels[1000] gives labels
            for k=1000.
        mlb: Fitted MultiLabelBinarizer mapping token strings to column indices
            in the sparse matrix. Useful for downstream interpretation of
            cluster gene content.
        results: List of dicts containing evaluation metrics per k value, each
            with keys: 'method', 'k', 'silhouette', 'calinski_harabasz',
            'davies_bouldin'. Ordered by the sequence of k_values evaluated.
    """
    t_pipeline = time.time()
    logger.info(
        f"Starting KMeans clustering pipeline for {len(modules):,} modules "
        f"(k_values={k_values}, use_svd={use_svd}, subsample_n={subsample_n})"
    )
    log_memory("pipeline start")

    # Step 0: Sparse matrix
    X, mlb = create_sparse_matrix(modules)

    # Optional subsample
    if subsample_n is not None and subsample_n < X.shape[0]:
        logger.info(f"Subsampling {X.shape[0]:,} → {subsample_n:,} modules...")
        idx = np.random.default_rng(42).choice(X.shape[0], size=subsample_n, replace=False)
        X   = X[idx]
        log_matrix_stats(X, "subsampled sparse matrix")

    # Step 1: Optional SVD
    if use_svd:
        X_input, n_components, cumvar = reduce_to_variance_threshold(
            X, target_variance=target_variance
        )
        del X
        gc.collect()
        log_memory("after dimensionality reduction")
        svd_label = f"SVD{n_components}"
        logger.info(
            f"Reduced dimensionality to {n_components} components, "
            f"preserving {cumvar[n_components-1]*100:.1f}% of variance"
        )
    else:
        logger.warning(
            "SVD disabled — clustering on raw sparse matrix. "
            "KMeans may converge in 1 iteration due to distance collapse."
        )
        X_input   = X
        svd_label = "noSVD"

    # Step 2: Evaluate k values
    logger.info(f"Evaluating {len(k_values)} k values: {k_values}")
    all_labels = {}
    results    = []

    if outdir is not None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

    # Compute UMAP once — reuse for all k values
    X_2d, umap_idx = None, None
    if outdir is not None:
        logger.info("Computing UMAP embedding once for all k values...")
        X_2d, umap_idx = compute_umap_embedding(X_input, n_jobs=n_jobs)
        log_memory("after UMAP embedding")
        gc.collect()

    for k in k_values:
        logger.info(f"Evaluating k={k} (batch_size=auto)...")
        method_label = f"KMeans (k={k}, {svd_label})"

        batch_size     = get_auto_batch_size(X_input.shape[0])
        labels, kmeans = minibatch_kmeans_clustering(
            X_input, k=k, batch_size=batch_size, n_init=n_init
        )
        all_labels[k] = labels

        eval_result = evaluate_kmeans(X_input, labels, method_label)
        if eval_result:
            results.append(eval_result)

        if outdir is not None:
            plot_cluster_report(
                labels, method_label,
                save_path=outdir / f'kmeans_report_k{k}_{svd_label}.png'
            )
            plot_umap_clusters(
                X_2d, umap_idx, labels, method_label,
                save_path=outdir / f'kmeans_umap_k{k}_{svd_label}.png',
            )

        # free per-k allocations
        del labels, kmeans

        gc.collect()
        log_memory(f"after k={k} cleanup")
        
    # Free UMAP embedding after all k values done
    del X_2d, umap_idx
    gc.collect()
    log_memory("after UMAP cleanup")

    # Step 3: Summary
    if results:
        import pandas as pd

        df      = pd.DataFrame(results)
        best    = df.loc[df['silhouette'].idxmax()]
        logger.info(
            f"Results across k values:\n{df.to_string(index=False)}\n"
            f"Best k by silhouette: k={best['k']:.0f} "
            f"(silhouette={best['silhouette']:.4f})"
        )

    if use_svd:
        del X_input
        gc.collect()

    elapsed = time.time() - t_pipeline
    log_memory("pipeline end")
    logger.info(
        f"KMeans pipeline completed in {elapsed:.1f}s ({elapsed/60:.1f} min) "
        f"for {len(k_values)} k values"
    )

    return all_labels, mlb, results

import numpy as np
import warnings
import logging
import gc
import time
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.cluster import HDBSCAN as SklearnHDBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import pairwise_distances
from collections import Counter
import matplotlib.pyplot as plt
import umap
import psutil
import os

logger = logging.getLogger(__name__)


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


def log_distance_stats(X, label="", n_sample=2000):
    n_sample = min(n_sample, X.shape[0])
    idx      = np.random.default_rng(42).choice(X.shape[0], size=n_sample, replace=False)
    X_sub    = X[idx]
    if hasattr(X_sub, 'toarray'):
        X_sub = X_sub.toarray()

    dists = pairwise_distances(X_sub, metric='euclidean')
    upper = dists[np.triu_indices(n_sample, k=1)]
    pcts  = np.percentile(upper, [1, 5, 10, 25, 50, 75, 90, 99])

    logger.info(
        f"[DISTANCES] {label} (euclidean on L2-normalized ≡ cosine, n_sample={n_sample}):\n"
        f"  std={upper.std():.4f}  range=[{upper.min():.4f}, {upper.max():.4f}]\n"
        f"   1pct={pcts[0]:.4f}   5pct={pcts[1]:.4f}  10pct={pcts[2]:.4f}\n"
        f"  25pct={pcts[3]:.4f}  50pct={pcts[4]:.4f}  75pct={pcts[5]:.4f}\n"
        f"  90pct={pcts[6]:.4f}  99pct={pcts[7]:.4f}"
    )
    return upper


def log_label_stats(labels, label="Labels"):
    mask       = labels != -1
    n_noise    = (~mask).sum()
    noise_pct  = n_noise / len(labels) * 100
    n_clusters = len(set(labels[mask])) if mask.sum() > 0 else 0

    if n_clusters > 0:
        sizes   = np.array(sorted(Counter(labels[mask]).values(), reverse=True))
        pcts    = np.percentile(sizes, [25, 50, 75, 90, 99])
        buckets = {
            '1-10':     (sizes <= 10).sum(),
            '11-50':    ((sizes > 10)  & (sizes <= 50)).sum(),
            '51-200':   ((sizes > 50)  & (sizes <= 200)).sum(),
            '201-1000': ((sizes > 200) & (sizes <= 1000)).sum(),
            '1000+':    (sizes > 1000).sum(),
        }
        logger.info(
            f"[CLUSTERS] {label}:\n"
            f"  total_points={len(labels):,}  n_clusters={n_clusters:,}  "
            f"noise={n_noise:,} ({noise_pct:.1f}%)\n"
            f"  size: min={sizes.min()}  max={sizes.max()}  "
            f"mean={sizes.mean():.1f}  median={np.median(sizes):.1f}  "
            f"std={sizes.std():.1f}\n"
            f"  percentiles: 25th={pcts[0]:.0f}  50th={pcts[1]:.0f}  "
            f"75th={pcts[2]:.0f}  90th={pcts[3]:.0f}  99th={pcts[4]:.0f}\n"
            f"  buckets: {buckets}"
        )
    else:
        logger.info(
            f"[CLUSTERS] {label}: 0 clusters, {n_noise:,} noise points"
        )

    return n_clusters, n_noise, noise_pct


# ─────────────────────────────────────────────────────────────
# 0. SPARSE MATRIX CREATION
# ─────────────────────────────────────────────────────────────

def create_sparse_matrix(modules):
    """
    Convert list of module token sets to a sparse binary matrix.

    Parameters
    ----------
    modules : list of iterables
        Each element is a collection of token strings for one module.
        e.g. [['geneA', 'geneB'], ['geneB', 'geneC'], ...]

    Returns
    -------
    X : scipy sparse matrix, shape (n_modules, n_unique_tokens)
    mlb : fitted MultiLabelBinarizer (kept for token→column mapping)
    """
    logger.info(f"[SPARSE] Creating sparse matrix from {len(modules):,} modules...")
    log_memory("before sparse matrix")
    t_start = time.time()

    mlb = MultiLabelBinarizer(sparse_output=True)
    X   = mlb.fit_transform(modules)
    elapsed = time.time() - t_start

    log_matrix_stats(X, "sparse binary matrix")
    log_memory("after sparse matrix")
    logger.info(
        f"[SPARSE] Created in {elapsed:.1f}s — "
        f"{len(mlb.classes_):,} unique tokens"
    )

    # Feature stats — distribution of tokens per module
    nnz_per_row = np.diff(X.indptr)
    pcts        = np.percentile(nnz_per_row, [1, 5, 25, 50, 75, 95, 99])
    logger.info(
        f"[SPARSE] Tokens per module:\n"
        f"  min={nnz_per_row.min()}  max={nnz_per_row.max()}  "
        f"mean={nnz_per_row.mean():.1f}  median={np.median(nnz_per_row):.1f}\n"
        f"  1pct={pcts[0]:.0f}  5pct={pcts[1]:.0f}  25pct={pcts[2]:.0f}  "
        f"50pct={pcts[3]:.0f}  75pct={pcts[4]:.0f}  95pct={pcts[5]:.0f}  "
        f"99pct={pcts[6]:.0f}"
    )

    return X, mlb


# ─────────────────────────────────────────────────────────────
# 1. DIMENSIONALITY REDUCTION
# ─────────────────────────────────────────────────────────────

def reduce_to_variance_threshold(
    X, target_variance=0.50, step=50, max_components=3000, random_state=42
):
    """
    Finds minimum n_components explaining target_variance.
    Output is L2-normalized — euclidean distance on output == cosine distance.
    """
    max_components = min(max_components, X.shape[1] - 1)
    step           = min(step, max_components)
    best_n         = None
    cumvar         = 0.0
    n              = step

    logger.info(
        f"[SVD] Searching for n_components explaining {target_variance*100:.0f}% "
        f"variance (step={step}, max={max_components})..."
    )
    log_memory("before SVD")
    t_start = time.time()

    while n <= max_components:
        svd    = TruncatedSVD(n_components=n, random_state=random_state)
        svd.fit(X)
        cumvar = svd.explained_variance_ratio_.sum()
        logger.info(f"[SVD]   n_components={n:>5}: {cumvar*100:.2f}% variance explained")

        if cumvar >= target_variance:
            cumvar_per_component = svd.explained_variance_ratio_.cumsum()
            best_n   = int((cumvar_per_component >= target_variance).argmax()) + 1
            achieved = cumvar_per_component[best_n - 1] * 100
            logger.info(
                f"[SVD] Target reached — refined to {best_n} components "
                f"({achieved:.2f}% variance)"
            )
            break
        n += step

    if best_n is None:
        best_n = min(n - step, max_components)
        logger.warning(
            f"[SVD] Could not reach {target_variance*100:.0f}% within "
            f"{max_components} components. Best: {cumvar*100:.2f}% at "
            f"n_components={best_n}."
        )

    logger.info(f"[SVD] Fitting final SVD with n_components={best_n}...")
    svd       = TruncatedSVD(n_components=best_n, random_state=random_state)
    X_reduced = svd.fit_transform(X).astype(np.float32)
    X_reduced = normalize(X_reduced, norm='l2')
    elapsed   = time.time() - t_start

    log_matrix_stats(X_reduced, f"SVD output (n_components={best_n}, L2-normalized)")
    log_memory("after SVD")
    logger.info(f"[SVD] Completed in {elapsed:.1f}s")
    log_distance_stats(X_reduced, label=f"SVD n={best_n} L2-normalized")

    return X_reduced, best_n, svd.explained_variance_ratio_.cumsum()


# ─────────────────────────────────────────────────────────────
# 2. CLUSTERING
# ─────────────────────────────────────────────────────────────

def cluster_hdbscan(
    X,
    min_cluster_size=20,
    cluster_selection_method='eom',
    cluster_selection_epsilon=0.0,
    leaf_size=40,
    n_jobs=-1,
):
    logger.info(
        f"[HDBSCAN] Starting clustering:\n"
        f"  n_samples={X.shape[0]:,}  n_features={X.shape[1]:,}\n"
        f"  min_cluster_size={min_cluster_size}\n"
        f"  cluster_selection_method={cluster_selection_method}\n"
        f"  cluster_selection_epsilon={cluster_selection_epsilon}\n"
        f"  metric=euclidean on L2-normalized (≡ cosine)\n"
        f"  algorithm=ball_tree  n_jobs={n_jobs}  leaf_size={leaf_size}"
    )
    log_memory("before HDBSCAN")

    clusterer = SklearnHDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
        metric='euclidean',
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=cluster_selection_epsilon,
        n_jobs=n_jobs,
        algorithm='ball_tree',
        leaf_size=leaf_size,
        copy=True,
    )

    t_start = time.time()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="sklearn")
        labels = clusterer.fit_predict(X)
    elapsed = time.time() - t_start

    log_memory("after HDBSCAN")
    logger.info(f"[HDBSCAN] Completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    log_label_stats(
        labels,
        label=f"HDBSCAN ({cluster_selection_method}, eps={cluster_selection_epsilon}, "
              f"mcs={min_cluster_size})"
    )

    return labels, clusterer


# ─────────────────────────────────────────────────────────────
# 3. EPSILON SCAN
# ─────────────────────────────────────────────────────────────

def scan_epsilon(
    X,
    min_cluster_size=20,
    cluster_selection_method='eom',
    epsilons=None,
    n_jobs=-1,
):
    if epsilons is None:
        epsilons = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

    import pandas as pd
    NUMEXPR_MAX_THREADS = min(16, os.cpu_count())  # to suppress numexpr warnings
    

    logger.info(
        f"[EPSILON SCAN] Scanning {len(epsilons)} epsilon values on "
        f"{X.shape[0]:,} points (min_cluster_size={min_cluster_size})..."
    )

    results = []
    for eps in epsilons:
        t_start = time.time()
        clusterer = SklearnHDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=1,
            metric='euclidean',
            cluster_selection_method=cluster_selection_method,
            cluster_selection_epsilon=eps,
            n_jobs=n_jobs,
            algorithm='ball_tree',
            copy=True,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning, module="sklearn")
            labels = clusterer.fit_predict(X)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_pct  = (labels == -1).sum() / len(labels) * 100
        elapsed    = time.time() - t_start
        results.append({
            'epsilon':    eps,
            'n_clusters': n_clusters,
            'noise_pct':  round(noise_pct, 1),
            'time_s':     round(elapsed, 1),
        })
        logger.info(
            f"[EPSILON SCAN]   epsilon={eps:.2f}: {n_clusters:>6} clusters, "
            f"{noise_pct:.1f}% noise ({elapsed:.1f}s)"
        )

    df = pd.DataFrame(results)
    logger.info(f"[EPSILON SCAN] Summary:\n{df.to_string(index=False)}")
    return df


# ─────────────────────────────────────────────────────────────
# 4. EVALUATION
# ─────────────────────────────────────────────────────────────

def evaluate_clustering(X, labels, method_name):
    mask       = labels != -1
    n_noise    = (~mask).sum()
    noise_pct  = n_noise / len(labels) * 100
    n_clusters = len(set(labels[mask]))

    if n_clusters < 2:
        logger.warning(f"[EVAL] {method_name}: fewer than 2 clusters — skipping")
        return None

    logger.info(
        f"[EVAL] Evaluating {method_name} "
        f"(excluding {n_noise:,} noise points = {noise_pct:.1f}%)..."
    )

    X_eval      = X[mask]
    lbl_eval    = labels[mask]
    sample_size = min(10_000, len(X_eval))

    t_start = time.time()
    sil = silhouette_score(
        X_eval, lbl_eval, metric='euclidean', sample_size=sample_size
    )
    db  = davies_bouldin_score(X_eval, lbl_eval)
    ch  = calinski_harabasz_score(X_eval, lbl_eval)
    elapsed = time.time() - t_start

    logger.info(
        f"[EVAL] {method_name} ({elapsed:.1f}s):\n"
        f"  n_clusters={n_clusters:,}  noise={n_noise:,} ({noise_pct:.1f}%)\n"
        f"  silhouette(euclidean/L2≡cosine)={sil:.4f}  "
        f"calinski_harabasz={ch:.4f}  davies_bouldin={db:.4f}\n"
        f"  interpretation:\n"
        f"    silhouette: "
        f"{'good (>0.2)' if sil > 0.2 else 'moderate (0-0.2)' if sil > 0 else 'poor (<0)'}\n"
        f"    davies_bouldin: "
        f"{'good (<1.0)' if db < 1.0 else 'moderate (1-2)' if db < 2.0 else 'poor (>2)'}"
    )

    return {
        'method': method_name, 'n_clusters': n_clusters,
        'noise_pct': noise_pct, 'silhouette': sil,
        'calinski_harabasz': ch, 'davies_bouldin': db,
    }


# ─────────────────────────────────────────────────────────────
# 5. VISUALIZATION
# ─────────────────────────────────────────────────────────────

def plot_cluster_report(label_set, method, save_path=None):
    is_noise     = label_set == -1
    clean_labels = label_set[~is_noise]
    counts       = Counter(clean_labels)
    sizes        = np.array(sorted(counts.values(), reverse=True))
    n_clusters   = len(counts)
    n_noise      = is_noise.sum()
    noise_pct    = n_noise / len(label_set) * 100

    if len(sizes) == 0:
        logger.warning(f"[REPORT] No clusters found for {method} — skipping")
        return

    percentiles = np.percentile(sizes, [25, 50, 75, 90, 99])
    fig, axes   = plt.subplots(1, 3, figsize=(18, 6))
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
        logger.info(f"[REPORT] Saved to {save_path}")
    plt.show()


def plot_umap_clusters(X_svd, labels, method, save_path=None, n_jobs=1):
    n_subsample  = min(50_000, len(X_svd))
    idx          = np.random.default_rng(42).choice(
                       len(X_svd), size=n_subsample, replace=False
                   )
    X_sub        = X_svd[idx]
    labels_sub   = labels[idx]

    logger.info(f"[UMAP] Running on {n_subsample:,} points...")
    log_memory("before UMAP")
    t_start = time.time()

    reducer = umap.UMAP(
        n_components=2,
        random_state=42 if n_jobs == 1 else None,
        n_neighbors=30,
        min_dist=0.1,
        n_jobs=n_jobs,
        metric='euclidean',
    )
    X_2d    = reducer.fit_transform(X_sub)
    elapsed = time.time() - t_start
    logger.info(f"[UMAP] Completed in {elapsed:.1f}s")
    log_memory("after UMAP")

    fig, ax      = plt.subplots(figsize=(10, 8))
    noise_mask   = labels_sub == -1
    cluster_mask = ~noise_mask

    ax.scatter(X_2d[noise_mask,    0], X_2d[noise_mask,    1],
               c='lightgrey', s=2, alpha=0.2, linewidths=0, label='Noise')
    scatter = ax.scatter(
        X_2d[cluster_mask, 0], X_2d[cluster_mask, 1],
        c=labels_sub[cluster_mask], cmap='tab20', s=2, alpha=0.5, linewidths=0
    )

    n_clusters = len(set(labels_sub[cluster_mask]))
    plt.colorbar(scatter, ax=ax, label='Cluster label')
    ax.set_title(f'{method} — UMAP (n={n_subsample:,}, {n_clusters} clusters shown)')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"[UMAP] Saved to {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────
# 6. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def run_clustering_pipeline(
    modules,                      # ← now takes raw modules, not X
    target_variance  = 0.50,
    min_cluster_size = 20,
    cluster_method   = 'eom',
    epsilon          = 0.0,
    n_jobs           = -1,
    outdir           = None,
    scan_epsilons    = False,
    subsample_n      = None,
):
    """
    Full clustering pipeline from raw module token lists to cluster labels.

    Parameters
    ----------
    modules          : list of iterables of token strings
                       e.g. [['geneA', 'geneB'], ['geneB', 'geneC'], ...]
    target_variance  : float, SVD variance threshold (default 0.50)
    min_cluster_size : int, HDBSCAN min cluster size (default 20)
    cluster_method   : str, 'eom' or 'leaf' (default 'eom')
    epsilon          : float, cluster_selection_epsilon (default 0.0)
    n_jobs           : int, -1 = all cores (default -1)
    outdir           : str or Path, save plots here (default None)
    scan_epsilons    : bool, scan epsilon values on subsample first
    subsample_n      : int or None, subsample for dev runs (default None)

    Returns
    -------
    labels : np.ndarray, cluster labels (-1 = noise)
    mlb    : fitted MultiLabelBinarizer (token → column index mapping)
    """
    t_pipeline = time.time()
    logger.info("=" * 60)
    logger.info("[PIPELINE] Starting clustering pipeline")
    logger.info(
        f"[PIPELINE] Parameters:\n"
        f"  n_modules={len(modules):,}\n"
        f"  target_variance={target_variance}\n"
        f"  min_cluster_size={min_cluster_size}\n"
        f"  cluster_method={cluster_method}\n"
        f"  epsilon={epsilon}\n"
        f"  metric=euclidean on L2-normalized (≡ cosine)\n"
        f"  n_jobs={n_jobs}  subsample_n={subsample_n}"
    )
    logger.info("=" * 60)
    log_memory("pipeline start")

    # Step 0: Create sparse matrix
    logger.info("[PIPELINE] Step 0/4: Creating sparse binary matrix")
    X, mlb = create_sparse_matrix(modules)

    # Optional subsample — applied to sparse matrix before SVD
    if subsample_n is not None and subsample_n < X.shape[0]:
        logger.info(f"[PIPELINE] Subsampling {X.shape[0]:,} → {subsample_n:,} points...")
        idx = np.random.default_rng(42).choice(X.shape[0], size=subsample_n, replace=False)
        X   = X[idx]
        log_matrix_stats(X, "subsampled sparse matrix")

    # Step 1: SVD + L2 normalize
    logger.info("[PIPELINE] Step 1/4: Dimensionality reduction + L2 normalization")
    X_reduced, n_components, cumvar = reduce_to_variance_threshold(
        X, target_variance=target_variance
    )
    del X
    gc.collect()
    log_memory("after del sparse matrix")

    # Step 2: Epsilon scan
    if scan_epsilons:
        logger.info("[PIPELINE] Step 2/4: Epsilon scan on 10k subsample")
        scan_idx = np.random.default_rng(42).choice(
            len(X_reduced), size=min(10_000, len(X_reduced)), replace=False
        )
        scan_epsilon(
            X_reduced[scan_idx],
            min_cluster_size=min_cluster_size,
            cluster_selection_method=cluster_method,
            n_jobs=n_jobs,
        )
    else:
        logger.info("[PIPELINE] Step 2/4: Epsilon scan skipped")

    # Step 3: Cluster
    logger.info("[PIPELINE] Step 3/4: HDBSCAN clustering")
    method_label = (
        f"HDBSCAN ({cluster_method}, eps={epsilon}, mcs={min_cluster_size})"
    )
    labels, clusterer = cluster_hdbscan(
        X_reduced,
        min_cluster_size=min_cluster_size,
        cluster_selection_method=cluster_method,
        cluster_selection_epsilon=epsilon,
        n_jobs=n_jobs,
    )

    # Step 4: Evaluate + plot
    logger.info("[PIPELINE] Step 4/4: Evaluation and visualization")
    eval_results = evaluate_clustering(X_reduced, labels, method_label)

    if outdir is not None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        plot_cluster_report(
            labels, method_label,
            save_path=outdir / f'cluster_report_mcs{min_cluster_size}_eps{epsilon}.png'
        )
        plot_umap_clusters(
            X_reduced, labels, method_label,
            save_path=outdir / f'cluster_umap_mcs{min_cluster_size}_eps{epsilon}.png',
            n_jobs=n_jobs,
        )

    del X_reduced
    gc.collect()

    elapsed = time.time() - t_pipeline
    log_memory("pipeline end")
    logger.info(
        f"[PIPELINE] Complete in {elapsed:.1f}s ({elapsed/60:.1f} min)\n"
        f"  final: "
        f"{eval_results['n_clusters'] if eval_results else 'N/A'} clusters, "
        f"{eval_results['noise_pct']:.1f}% noise" if eval_results else ""
    )
    logger.info("=" * 60)

    return labels, n_components, mlb

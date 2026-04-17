"""
Visualization module for clustering comparison results.
"""

import matplotlib
matplotlib.use('Agg')  # MUST come before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import logging
from pathlib import Path
from collections import Counter
import warnings

logger = logging.getLogger(__name__)

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logger.warning("UMAP not available. Install with: pip install umap-learn")


def plot_cluster_report(labels, method_name, save_path=None):
    """
    Generate cluster size distribution report.
    
    Args:
        labels: Cluster labels (-1 = noise)
        method_name: Name for plot title
        save_path: Path to save plot (optional)
    """
    is_noise     = labels == -1
    clean_labels = labels[~is_noise]
    counts       = Counter(clean_labels)
    sizes        = np.array(sorted(counts.values(), reverse=True))
    n_clusters   = len(counts)
    n_noise      = is_noise.sum()
    noise_pct    = n_noise / len(labels) * 100

    if len(sizes) == 0:
        logger.warning(f"[PLOT] No clusters found for {method_name} — skipping")
        return

    percentiles = np.percentile(sizes, [25, 50, 75, 90, 99])
    fig, axes   = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Cluster Size Report — {method_name}', fontsize=14, fontweight='bold')

    # Histogram
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
    ax.grid(axis='y', alpha=0.3)

    # Ranked sizes
    ax = axes[1]
    ax.plot(range(len(sizes)), sizes, color='steelblue', linewidth=1)
    ax.fill_between(range(len(sizes)), sizes, alpha=0.3, color='steelblue')
    ax.set_title('Ranked cluster sizes')
    ax.set_xlabel('Cluster rank (largest → smallest)')
    ax.set_ylabel('Cluster size (n points)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Statistics table
    ax = axes[2]
    ax.axis('off')
    stats_text = (
        f"Method: {method_name}\n"
        f"{'─' * 28}\n"
        f"Total points:     {len(labels):>10,}\n"
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
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"[PLOT] Saved cluster report to {save_path}")
    plt.close()


def plot_umap_projection(X_reduced, labels, method_name, save_path=None, 
                         n_subsample=50000, n_jobs=1):
    """
    Generate UMAP 2D projection of clusters.
    
    Args:
        X_reduced: SVD-reduced matrix (or raw if no SVD)
        labels: Cluster labels (-1 = noise)
        method_name: Name for plot title
        save_path: Path to save plot (optional)
        n_subsample: Maximum points to plot (default 50,000)
        n_jobs: Number of parallel jobs for UMAP
    """
    if not UMAP_AVAILABLE:
        logger.warning("[PLOT] UMAP not available - skipping projection plot")
        return
    
    n_subsample  = min(n_subsample, len(X_reduced))
    idx          = np.random.default_rng(42).choice(
                       len(X_reduced), size=n_subsample, replace=False
                   )
    X_sub        = X_reduced[idx]
    labels_sub   = labels[idx]

    logger.info(f"[PLOT] Computing UMAP projection on {n_subsample:,} points...")
    
    reducer = umap.UMAP(
        n_components=2,
        random_state=42 if n_jobs == 1 else None,
        n_neighbors=30,
        min_dist=0.1,
        n_jobs=n_jobs,
        metric='euclidean',
    )
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        X_2d = reducer.fit_transform(X_sub)
    
    logger.info(f"[PLOT] UMAP projection complete")

    # Plot
    fig, ax      = plt.subplots(figsize=(10, 8))
    noise_mask   = labels_sub == -1
    cluster_mask = ~noise_mask

    # Plot noise points
    ax.scatter(X_2d[noise_mask, 0], X_2d[noise_mask, 1],
               c='lightgrey', s=2, alpha=0.2, linewidths=0, label='Noise')
    
    # Plot clusters
    if cluster_mask.sum() > 0:
        scatter = ax.scatter(
            X_2d[cluster_mask, 0], X_2d[cluster_mask, 1],
            c=labels_sub[cluster_mask], cmap='tab20', s=1.5, alpha=0.3, linewidths=0
        )
        n_clusters = len(set(labels_sub[cluster_mask]))
        plt.colorbar(scatter, ax=ax, label='Cluster label')
    else:
        n_clusters = 0

    ax.set_title(f'{method_name} — UMAP Projection (n={n_subsample:,}, {n_clusters} clusters)')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"[PLOT] Saved UMAP projection to {save_path}")
    plt.close()


def generate_clustering_plots(config_name, X_reduced, labels, metadata, output_dir):
    """
    Generate all plots for a single configuration.
    
    Args:
        config_name: Name of configuration
        X_reduced: Reduced matrix (after SVD)
        labels: Cluster labels
        metadata: Dict with clustering metadata
        output_dir: Directory to save plots
    """
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    method_label = f"{config_name} ({metadata.get('method', 'unknown')})"
    
    logger.info(f"[PLOT] Generating plots for {config_name}...")
    
    # Cluster size report
    plot_cluster_report(
        labels, 
        method_label,
        save_path=plots_dir / "cluster_size_report.png"
    )
    
    # UMAP projection
    plot_umap_projection(
        X_reduced,
        labels,
        method_label,
        save_path=plots_dir / "umap_projection.png",
        n_subsample=50000,
        n_jobs=-1
    )
    
    logger.info(f"[PLOT] All plots saved to {plots_dir}")


def plot_evaluation_summary(results: dict, save_path: Path):
    """
    Generate streamlined evaluation summary plot.
    
    Args:
        results: Dict of {config_name: results with all pipeline stages and scores}
        save_path: Path to save plot
    """
    if not results:
        logger.warning("[PLOT] No results to plot")
        return
    
    # Filter out errors
    valid_results = [(name, r) for name, r in results.items() if 'error' not in r]
    
    if not valid_results:
        logger.warning("[PLOT] No valid results to plot")
        return
    
    # Keep original order (UNSORTED)
    names = [name for name, _ in valid_results]
    
    # Extract all metrics
    n_initial = [r['n_initial_motifs'] for _, r in valid_results]
    n_merged = [r['n_merged_motifs'] for _, r in valid_results]
    n_gwms = [r['n_gwms'] for _, r in valid_results]
    overlap_scores = [r['mean_overlap_score'] for _, r in valid_results]
    mrpos_scores = [r['mean_penalized_score'] for _, r in valid_results]
    
    # Extract metadata for cluster quality
    methods = [r['config']['method'].upper() for _, r in valid_results]
    silhouette = [r.get('metadata', {}).get('silhouette_score', np.nan) for _, r in valid_results]
    davies_bouldin = [r.get('metadata', {}).get('davies_bouldin_score', np.nan) for _, r in valid_results]
    calinski = [r.get('metadata', {}).get('calinski_harabasz_score', np.nan) for _, r in valid_results]
    noise_fraction = [r.get('metadata', {}).get('noise_fraction', np.nan) * 100 for _, r in valid_results]
    
    # Calculate derived metrics for summary table
    merge_reduction = [((ni - nm) / ni * 100) if ni > 0 else 0 
                       for ni, nm in zip(n_initial, n_merged)]
    gwm_success = [(ng / nm * 100) if nm > 0 else 0 
                   for ng, nm in zip(n_gwms, n_merged)]
    overall_yield = [(ng / ni * 100) if ni > 0 else 0 
                     for ng, ni in zip(n_gwms, n_initial)]
    
    # Method colors
    method_colors = {'KMEANS': '#3498db', 'HDBSCAN': '#9b59b6'}
    colors = [method_colors.get(m, '#95a5a6') for m in methods]
    
    # Create figure with 2x2 layout
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.35,
                         height_ratios=[1.2, 1])
    
    fig.suptitle('ClusterClue GWM Evaluation Summary', 
                 fontsize=16, fontweight='bold', y=0.97)
    
    x_pos = np.arange(len(names))
    
    # ============================================================================
    # Plot 1: Pipeline Attrition (Top Left)
    # ============================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    merged_lost = np.array(n_initial) - np.array(n_merged)
    gwm_lost = np.array(n_merged) - np.array(n_gwms)
    
    ax1.bar(x_pos, n_gwms, label='Final GWMs', color='#27ae60', alpha=0.85, 
            edgecolor='black', linewidth=0.5)
    ax1.bar(x_pos, gwm_lost, bottom=n_gwms, label='Lost in GWM filtering', 
            color='#e67e22', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.bar(x_pos, merged_lost, bottom=np.array(n_merged), 
            label='Merged away', color='#95a5a6', alpha=0.5, 
            edgecolor='black', linewidth=0.5)
    
    ax1.set_xlabel('Configuration', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Number of Motifs', fontsize=11, fontweight='bold')
    ax1.set_title('Pipeline Attrition: Initial → Merged → Final GWMs', 
                  fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax1.legend(loc='upper right', fontsize=9, framealpha=0.95)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add overall yield % on top
    for i, (x, total, final) in enumerate(zip(x_pos, n_initial, n_gwms)):
        yield_pct = (final / total * 100) if total > 0 else 0
        ax1.text(x, total + max(n_initial)*0.02, f'{yield_pct:.0f}%', 
                ha='center', va='bottom', fontsize=9, fontweight='bold', 
                color='#27ae60')
    
    # ============================================================================
    # Plot 2: Biological Validation Scores (Top Right)
    # ============================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Dual bar chart
    width = 0.35
    x = np.arange(len(names))
    
    bars1 = ax2.bar(x - width/2, overlap_scores, width, 
                    label='Overlap Score', color='#3498db', alpha=0.8, 
                    edgecolor='black', linewidth=0.5)
    bars2 = ax2.bar(x + width/2, mrpos_scores, width, 
                    label='MRPOS Score', color='#e74c3c', alpha=0.8, 
                    edgecolor='black', linewidth=0.5)
    
    ax2.set_xlabel('Configuration', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax2.set_title('Biological Validation Scores', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax2.legend(fontsize=9, loc='upper right', framealpha=0.95)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=7)
    
    # ============================================================================
    # Plot 3: Cluster Quality Metrics Heatmap (Bottom Left)
    # ============================================================================
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Prepare heatmap data
    # Normalize metrics to 0-1 scale for visualization
    metrics_data = []
    metric_labels = []
    
    # Silhouette: -1 to 1 → normalize to 0-1
    if not all(np.isnan(silhouette)):
        sil_norm = [(s + 1) / 2 if not np.isnan(s) else 0 for s in silhouette]
        metrics_data.append(sil_norm)
        metric_labels.append('Silhouette\n(higher better)')
    
    # Davies-Bouldin: lower is better, so invert
    # Typical range 0-3, invert and normalize
    if not all(np.isnan(davies_bouldin)):
        db_max = max([db for db in davies_bouldin if not np.isnan(db)]) if any(not np.isnan(db) for db in davies_bouldin) else 3
        db_norm = [1 - (db / (db_max * 1.2)) if not np.isnan(db) else 0 for db in davies_bouldin]
        metrics_data.append(db_norm)
        metric_labels.append('Davies-Bouldin\n(lower better)')
    
    # Calinski-Harabasz: higher is better, normalize to max
    if not all(np.isnan(calinski)):
        ch_max = max([ch for ch in calinski if not np.isnan(ch)]) if any(not np.isnan(ch) for ch in calinski) else 1
        ch_norm = [(ch / ch_max) if not np.isnan(ch) else 0 for ch in calinski]
        metrics_data.append(ch_norm)
        metric_labels.append('Calinski-Harabasz\n(higher better)')
    
    # Noise fraction: lower is better, so invert
    if not all(np.isnan(noise_fraction)):
        nf_norm = [1 - (nf / 100) if not np.isnan(nf) else 0 for nf in noise_fraction]
        metrics_data.append(nf_norm)
        metric_labels.append('Low Noise\n(lower better)')
    
    # Create heatmap
    if metrics_data:
        heatmap_data = np.array(metrics_data)
        
        im = ax3.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks
        ax3.set_xticks(np.arange(len(names)))
        ax3.set_yticks(np.arange(len(metric_labels)))
        ax3.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
        ax3.set_yticklabels(metric_labels, fontsize=9)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        cbar.set_label('Normalized Quality\n(0=worst, 1=best)', 
                      rotation=270, labelpad=20, fontsize=9)
        
        # Add text annotations
        for i in range(len(metric_labels)):
            for j in range(len(names)):
                text = ax3.text(j, i, f'{heatmap_data[i, j]:.2f}',
                              ha="center", va="center", color="black", 
                              fontsize=7, fontweight='bold')
        
        ax3.set_title('Cluster Quality Metrics Heatmap\n(normalized for comparison)', 
                     fontsize=12, fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'No cluster quality metrics available', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=11)
        ax3.set_title('Cluster Quality Metrics', fontsize=12, fontweight='bold')
    
    # ============================================================================
    # Plot 4: Summary Statistics Table (Bottom Right)
    # ============================================================================
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    # Find best configurations
    best_mrpos_idx = np.argmax(mrpos_scores)
    best_overlap_idx = np.argmax(overlap_scores)
    best_gwms_idx = np.argmax(n_gwms)
    best_yield_idx = np.argmax(overall_yield)
    best_sil_idx = np.nanargmax(silhouette) if not all(np.isnan(silhouette)) else 0
    
    # Calculate averages
    avg_initial = np.mean(n_initial)
    avg_merged = np.mean(n_merged)
    avg_gwms = np.mean(n_gwms)
    avg_merge_reduction = np.mean(merge_reduction)
    avg_gwm_success = np.mean(gwm_success)
    avg_overall_yield = np.mean(overall_yield)
    avg_overlap = np.mean(overlap_scores)
    avg_mrpos = np.mean(mrpos_scores)
    avg_sil = np.nanmean(silhouette)
    
    summary_text = (
        f"{'SUMMARY STATISTICS':^70}\n"
        f"{'═' * 70}\n\n"
        f"{'BEST CONFIGURATIONS':^70}\n"
        f"{'─' * 70}\n"
        f"Best MRPOS Score:\n"
        f"  → {names[best_mrpos_idx][:50]}\n"
        f"  → Score: {mrpos_scores[best_mrpos_idx]:.4f}\n\n"
        f"Best Overlap Score:\n"
        f"  → {names[best_overlap_idx][:50]}\n"
        f"  → Score: {overlap_scores[best_overlap_idx]:.4f}\n\n"
        f"Most GWMs Produced:\n"
        f"  → {names[best_gwms_idx][:50]}\n"
        f"  → GWMs: {n_gwms[best_gwms_idx]:,}\n\n"
        f"Best Overall Yield:\n"
        f"  → {names[best_yield_idx][:50]}\n"
        f"  → Yield: {overall_yield[best_yield_idx]:.1f}%\n\n"
        f"{'─' * 70}\n"
        f"{'AVERAGE METRICS':^70}\n"
        f"{'─' * 70}\n"
        f"Pipeline:\n"
        f"  Initial motifs:          {avg_initial:>8.0f}\n"
        f"  Merged motifs:           {avg_merged:>8.0f}\n"
        f"  Final GWMs:              {avg_gwms:>8.0f}\n"
        f"  Merging reduction:       {avg_merge_reduction:>7.1f}%\n"
        f"  GWM build success:       {avg_gwm_success:>7.1f}%\n"
        f"  Overall yield:           {avg_overall_yield:>7.1f}%\n\n"
        f"Biological Validation:\n"
        f"  Overlap score:           {avg_overlap:>8.4f}\n"
        f"  MRPOS score:             {avg_mrpos:>8.4f}\n\n"
        f"Cluster Quality:\n"
        f"  Silhouette:              {avg_sil:>8.4f}\n\n"
        f"{'═' * 70}\n"
    )
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.9, 
                     edgecolor='#34495e', linewidth=2))
    
    # Add method legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label='K-means', alpha=0.8, edgecolor='black'),
        Patch(facecolor='#9b59b6', label='HDBSCAN', alpha=0.8, edgecolor='black'),
    ]
    ax4.legend(handles=legend_elements, loc='lower right', 
              fontsize=9, title='Clustering Method', title_fontsize=10, 
              framealpha=0.95)
    
    # Save
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    logger.info(f"[PLOT] Saved evaluation summary to {save_path}")
    plt.close()
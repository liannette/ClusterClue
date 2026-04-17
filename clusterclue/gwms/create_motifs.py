import logging
from collections import Counter
from itertools import product
from pathlib import Path
from collections import defaultdict
from typing import Dict, DefaultDict, List
import numpy as np
import pandas as pd
import pickle
import json
import gc

from clusterclue.classes.subcluster_motif import SubclusterMotif
from clusterclue.clusters.utils import read_clusters
from clusterclue.gwms.create.build_gwms import build_motif_gwms, write_motif_gwms
from clusterclue.gwms.create.clustering import ClusteringComparison
from clusterclue.gwms.create.combine_matches import combine_presto_matches
from clusterclue.gwms.create.merge_motifs import merge_similar_motifs
from clusterclue.gwms.create.plots import generate_clustering_plots, plot_evaluation_summary
from clusterclue.evaluate.evaluate_hits import (
    assign_best_hit,
    read_reference_subclusters_and_tokenize_genes
)
from clusterclue.gwms.detect_motifs import (
    detect_motifs, write_motif_hits, parse_clusters_file, parse_motifs_file
)


logger = logging.getLogger(__name__)


def get_gene_background_count(clusters: dict) -> Counter:
    """Counts how many BGCs each tokenised gene occurs in."""
    gene_counts = Counter()
    for genes in clusters.values():
        tokenized_genes = set([';'.join(gene) for gene in genes])
        gene_counts.update(tokenized_genes)
    # remove genes without biosynthetic domains
    gene_counts.pop("-", None) 
    return gene_counts


def generate_evaluation_report(results: dict, output_file: Path):
    """Generate human-readable biological evaluation report."""
    with open(output_file, 'w') as f:
        f.write("="*120 + "\n")
        f.write("GWM EVALUATION REPORT\n")
        f.write("="*120 + "\n\n")
        
        f.write("This report compares different configurations based on their ability to\n")
        f.write("produce biologically valid Gene Weight Modules (GWMs) that match known subclusters.\n\n")
        
        # Summary table with all important metrics
        f.write("SUMMARY TABLE\n")
        f.write("-"*120 + "\n")
        f.write(f"{'Config':<30} {'Method':<10} {'Initial':<8} {'Merged':<8} {'GWMs':<8} "
                f"{'Overlap':<10} {'MRPOS':<10}\n")
        f.write(f"{'':30} {'':10} {'Motifs':8} {'Motifs':8} {'':8} {'Score':10} {'Score':10}\n")
        f.write("-"*120 + "\n")
        
        # Sort by penalized score (MRPOS - most important metric)
        sorted_results = sorted(
            [(name, r) for name, r in results.items() if 'error' not in r],
            key=lambda x: x[1]['mean_penalized_score'],
            reverse=True
        )
        
        for name, result in sorted_results:
            method = result['config']['method'].upper()
            n_initial = result['n_initial_motifs']
            n_merged = result['n_merged_motifs']
            n_gwms = result['n_gwms']
            overlap = result['mean_overlap_score']
            penalized = result['mean_penalized_score']
            
            f.write(f"{name:<30} {method:<10} {n_initial:<8} {n_merged:<8} {n_gwms:<8} "
                    f"{overlap:<10.4f} {penalized:<10.4f}\n")
        
        f.write("-"*120 + "\n\n")
    
    
        # Winner section
        winner_name, winner = sorted_results[0]
        f.write("="*120 + "\n")
        f.write("BEST CONFIGURATION\n")
        f.write("="*120 + "\n\n")
        f.write(f"Winner: {winner_name}\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  Method: {winner['config']['method'].upper()}\n")
        
        if winner['config']['method'] == 'kmeans':
            f.write(f"  K: {winner['config'].get('k', 'N/A')}\n")
        else:
            f.write(f"  Min cluster size: {winner['config'].get('min_cluster_size', 'N/A')}\n")
            f.write(f"  Cluster selection epsilon: {winner['config'].get('cluster_selection_epsilon', 'N/A')}\n")
        
        f.write(f"  SVD: {winner['config'].get('use_svd', False)}\n")
        if winner['config'].get('use_svd', False):
            f.write(f"  Target variance: {winner['config'].get('target_variance', 'N/A')}\n")
        
        f.write(f"\nGWM Hyperparameters:\n")
        params = winner['gwm_hyperparameter']
        f.write(f"  Min matches: {params['min_matches']}\n")
        f.write(f"  Min core genes: {params['min_core_genes']}\n")
        f.write(f"  Core threshold: {params['core_threshold']}\n")
        f.write(f"  Min gene probability: {params['min_gene_prob']}\n")
        
        f.write(f"\nPipeline Results:\n")
        f.write(f"  Initial motifs: {winner['n_initial_motifs']}\n")
        f.write(f"  Merged motifs: {winner['n_merged_motifs']} "
                f"({(winner['n_initial_motifs']-winner['n_merged_motifs'])/winner['n_initial_motifs']*100:.1f}% reduction)\n")
        f.write(f"  Final GWMs: {winner['n_gwms']} "
                f"({winner['n_gwms']/winner['n_merged_motifs']*100:.1f}% build success)\n")
        
        f.write(f"\nBiological Validation:\n")
        f.write(f"  Mean overlap score: {winner['mean_overlap_score']:.4f}\n")
        f.write(f"  Mean MRPOS (penalized): {winner['mean_penalized_score']:.4f}\n")
        
        f.write(f"\nCluster Quality Metrics:\n")
        metadata = winner.get('metadata', {})
        if 'silhouette_score' in metadata:
            f.write(f"  Silhouette score: {metadata['silhouette_score']:.4f}\n")
        if 'davies_bouldin_score' in metadata:
            f.write(f"  Davies-Bouldin score: {metadata['davies_bouldin_score']:.4f}\n")
        if 'calinski_harabasz_score' in metadata:
            f.write(f"  Calinski-Harabasz score: {metadata['calinski_harabasz_score']:.2f}\n")
        if 'noise_fraction' in metadata:
            f.write(f"  Noise fraction: {metadata['noise_fraction']:.2%}\n")
        
        f.write("\n")
        
        f.write("="*120 + "\n")
        f.write("\nKEY METRICS EXPLAINED\n")
        f.write("-"*120 + "\n")
        f.write("Overlap Score:      How well GWMs match reference subclusters (F1 score)\n")
        f.write("MRPOS Score:        Mean Redundancy Penalized Overlap Score - overlap with penalty for cluster size\n")
        f.write("Silhouette Score:   Cluster separation quality (-1 to 1, higher is better)\n")
        f.write("Davies-Bouldin:     Cluster compactness (lower is better)\n")
        f.write("Calinski-Harabasz:  Cluster definition quality (higher is better)\n")
        f.write("Noise Fraction:     Proportion of modules classified as noise (HDBSCAN only)\n")
        f.write("="*120 + "\n")


def save_module_label_mapping(modules: List[str], labels: np.ndarray, filepath: Path):
    """Save module to cluster label mapping."""
    mapping = pd.DataFrame({
        'module': [','.join(mod) for mod in modules],
        'cluster_label': labels
    })
    mapping.to_csv(filepath, sep='\t', index=False)
    logger.info(f"Saved module-label mapping to {filepath}")
    return mapping


def analyze_motifs(motifs: Dict[str, SubclusterMotif]) -> Dict:
    """Analyze motif statistics."""
    if not motifs:
        return {}
    
    match_counts = [m.n_matches for m in motifs.values()]
    core_gene_counts = [len(m.core_genes) for m in motifs.values()]
    all_genes = set()
    for m in motifs.values():
        all_genes.update(m.tokenized_genes)
    
    return {
        'total_motifs': len(motifs),
        'total_unique_genes': len(all_genes),
        'match_count_mean': np.mean(match_counts),
        'match_count_median': np.median(match_counts),
        'match_count_std': np.std(match_counts),
        'match_count_min': np.min(match_counts),
        'match_count_max': np.max(match_counts),
        'core_genes_mean': np.mean(core_gene_counts),
        'core_genes_median': np.median(core_gene_counts),
        'core_genes_std': np.std(core_gene_counts),
        'core_genes_min': np.min(core_gene_counts),
        'core_genes_max': np.max(core_gene_counts),
    }


def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def generate_subcluster_motifs(
    configs: List[dict], 
    stat_matches_filepath: Path,
    top_matches_filepath: Path,
    out_dirpath: Path,
    n_jobs: int
    ):
    logger.info("=== Generating subcluster motifs ===")

    # Use TOP and STAT results to create subcluster motifs
    combined_matches = combine_presto_matches(
        stat_matches_filepath, 
        top_matches_filepath,
    )
    
    # create mapping for subcluster module to bgcs
    module2bgcs = defaultdict(list)
    for bgc_id, module in combined_matches:
        module2bgcs[module].append(bgc_id)
    modules = sorted(module2bgcs.keys())

    logger.info(f"Loaded {len(modules)} unique modules")

    # Initialize comparison
    comparison = ClusteringComparison(modules, module2bgcs)

    results = {}

    for config in configs:
        name = config['name']
        logger.info(f"Running clustering with configuration '{name}'...")

        # create config-specific output directory
        config_dir = out_dirpath / name
        config_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Step 1: Cluster
            if config['method'] == 'kmeans':
                labels, metadata = comparison.run_kmeans(
                    k=config['k'],
                    use_svd=config.get('use_svd', False),
                    target_variance=config.get('target_variance', 0.30)
                )
            elif config['method'] == 'hdbscan':
                labels, metadata = comparison.run_hdbscan(
                    min_cluster_size=config['min_cluster_size'],
                    use_svd=config.get('use_svd', True),
                    target_variance=config.get('target_variance', 0.30),
                    min_samples=config.get('min_samples', None),
                    cluster_selection_epsilon=config.get('cluster_selection_epsilon', 0.0),
                    n_jobs=n_jobs
                )
            else:
                logger.error(f"Unknown method: {config['method']}")
                continue

            # Save module-label mapping
            save_module_label_mapping(
                modules, 
                labels, 
                config_dir / "module_labels.tsv"
            )

            # Calculate cluster quality metrics
            cluster_metrics = comparison.calculate_cluster_metrics(labels)
            metadata.update(cluster_metrics)

            # Convert to motifs
            motifs = comparison.labels_to_motifs(labels)
            logger.info(f"Created {len(motifs)} initial motifs")
            
            # save as pkl 
            motifs_filepath = config_dir / f"motifs_{name}.pkl"
            with open(motifs_filepath, "wb") as f:
                pickle.dump(motifs, f)
            logger.info(f"Saved initial motifs to {motifs_filepath}")

            # Merge similar motifs
            merge_similarity_threshold = 0.7
            merge_gene_threshold = 0.2
            similarity_metric = "jaccard"
            logger.info(
                f"Merging motifs with {similarity_metric} similarity >= "
                f"{merge_similarity_threshold} (gene_prob >= {merge_gene_threshold})..."
            )
            merged_motifs = merge_similar_motifs(
                motifs, 
                similarity_threshold=merge_similarity_threshold,
                gene_prob_threshold=merge_gene_threshold,
                similarity_metric=similarity_metric
            )
            logger.info(f"Reduced {len(motifs)} motifs to {len(merged_motifs)} motifs after merging")

            # save merged motifs to file
            merged_motifs_filepath = config_dir / f"motifs_merged_{name}.pkl"
            with open(merged_motifs_filepath, "wb") as f:
                pickle.dump(merged_motifs, f)
            logger.info(f"Saved merged motifs to {merged_motifs_filepath}")

            # Store clustering results
            results[name] = {
                'config': config,
                'metadata': metadata,
                'n_initial_motifs': len(motifs),
                'n_merged_motifs': len(merged_motifs),
                'motifs_filepath': str(motifs_filepath),
                'merged_motifs_filepath': str(merged_motifs_filepath),
            }
        
        except Exception as e:
            logger.error(f"Error in config {name}: {e}", exc_info=True)
            results[name] = {'error': str(e)}

        # Save results after each config to avoid losing data if something crashes
        summary_file = out_dirpath / "results_clustering.json"
        with open(summary_file, 'w') as f:
            json.dump(convert_numpy_types(results), f, indent=2)

        # Generate plots
        try:
            # Get reduced matrix for plotting
            if metadata.get('use_svd', False):
                # Use cached SVD result
                X_plot, _, _ = comparison.apply_svd(
                    metadata.get('target_variance', 0.30)
                )
            else:
                # Use raw sparse matrix (convert to dense for plotting)
                X_plot = comparison.X.toarray()
            
            # Generate plots
            generate_clustering_plots(name, X_plot, labels, metadata, config_dir)
        except Exception as e:
            logger.warning(f"Error generating plots for {name}: {e}")
    
    return results


def generate_subcluster_gwms(
    clustering_results: dict, 
    gwm_hyperparams,
    clusters_filepath: Path, 
    out_dirpath: Path
    ):
    logger.info("=== Building GWMs for merged motifs ===")
    
    gwms_dirpath = out_dirpath / "gwms"
    gwms_dirpath.mkdir(parents=True, exist_ok=True)

    # some variables necessary to build the gwms
    training_clusters = read_clusters(clusters_filepath)
    n_clusters_total = len(training_clusters)
    background_counts = get_gene_background_count(training_clusters)

    new_results = {}
    
    for config_name, config_results in clustering_results.items():
        if 'error' in config_results:
            logger.warning(f"Skipping GWM building for {config_name} due to previous error: {config_results['error']}")
            continue

        # open the merged motifs file
        merged_motifs_filepath = config_results.get('merged_motifs_filepath')
        with open(merged_motifs_filepath, "rb") as f:
            merged_motifs = pickle.load(f)
        logger.info(f"Loaded {len(merged_motifs)} merged motifs for {config_name}")

        for mm, mgc, ct, mgp in gwm_hyperparams:
            params_str = f"mm{mm}_mgc{mgc}_ct{int(ct * 100)}_mgp{int(mgp * 100)}"
            name = f"{config_name}_{params_str}"

            logger.info(f"Building GWMs for {name}...")
                
            gwms = build_motif_gwms(
                merged_motifs,
                background_counts,
                n_clusters_total,
                min_matches=mm,
                min_core_genes=mgc,
                core_threshold=ct,
                min_gene_prob=mgp,
            )
            logger.info(f"Final GWMs: {len(gwms)}")

            gwm_filepath = gwms_dirpath / f"GWMs_{name}.txt"
            write_motif_gwms(gwms, gwm_filepath)

            gwms_stats = analyze_motifs(gwms)
        
            # Store gwm results
            new_results[name] = {
                'config': config_results['config'],
                'metadata': config_results['metadata'],
                'n_initial_motifs': config_results['n_initial_motifs'],
                'n_merged_motifs': config_results['n_merged_motifs'],
                'motifs_filepath': config_results['motifs_filepath'],
                'merged_motifs_filepath': config_results['merged_motifs_filepath'],
                'gwm_hyperparameter': {
                    'min_matches': mm,
                    'min_core_genes': mgc,
                    'core_threshold': ct,
                    'min_gene_prob': mgp,
                },
                'n_gwms': len(gwms),
                'gwms_stats': gwms_stats,
                'gwms_filepath': str(gwm_filepath),
            }

            # Save results to avoid losing data if something crashes
            summary_file = out_dirpath / "results_gwms.json"
            with open(summary_file, 'w') as f:
                json.dump(convert_numpy_types(new_results), f, indent=2)
        
    return new_results


def evaluate_subcluster_gwms(
    results: dict,
    annotated_subclusters_filepath: Path,
    reference_clusters_filepath: Path,
    overlap_penalty_alpha: float,
    overlap_penalty_beta: float,
    out_dirpath: Path
    ):
    logger.info("=== Evaluating GWMs ===")
    
    annotated_subclusters = read_reference_subclusters_and_tokenize_genes(
        annotated_subclusters_filepath, 
        reference_clusters_filepath.parent / "all_domain_hits.txt"
    )
    logger.info(f"Loaded {len(annotated_subclusters)} annotated subclusters")

    ref_clusters = parse_clusters_file(reference_clusters_filepath)
    logger.info(f"Loaded {len(ref_clusters)} reference clusters")

    hits_dirpath = out_dirpath / "evaluation_hits"
    hits_dirpath.mkdir(parents=True, exist_ok=True)

    evaluation_dirpath = out_dirpath / "evaluation_best_hits"
    evaluation_dirpath.mkdir(parents=True, exist_ok=True)

    for gwm_name, gwm_results in results.items():
        
        logger.info(f"Evaluating GWMs for {gwm_name}...")

        # Detect motifs
        gwm_filepath = Path(gwm_results['gwms_filepath'])
        gwms = parse_motifs_file(gwm_filepath)
        hits = detect_motifs(ref_clusters, gwms)

        logger.info(f"Found {len(hits)} motif hits")

        # Save motif hits to file            
        eval_hits_filepath = hits_dirpath / f"hits_{gwm_name}.txt"
        write_motif_hits(hits, gwms, eval_hits_filepath)

        # Evaluate motif hits against annotated subclusters
        eval_df = pd.DataFrame(
            annotated_subclusters.apply(
                lambda row: assign_best_hit(
                    row, hits, 
                    alpha=overlap_penalty_alpha, beta=overlap_penalty_beta
                ), axis=1
            ).tolist()
        )
        # Save to csv
        eval_best_hits_filepath = evaluation_dirpath / f"best_hits_{gwm_name}.txt"
        eval_df.to_csv(eval_best_hits_filepath, sep="\t", index=False)

        mean_overlap = eval_df["overlap_score"].mean()
        mean_penalized = eval_df["penalized_score"].mean()
        
        logger.info(f"Overlap score: {mean_overlap:.4f}")
        logger.info(f"Penalized score (alpha={overlap_penalty_alpha}, beta={overlap_penalty_beta}): {mean_penalized:.4f}")
        
        # Store results
        results[gwm_name].update({
            'mean_overlap_score': float(mean_overlap),
            'mean_penalized_score': float(mean_penalized),
            'alpha': overlap_penalty_alpha,
            'beta': overlap_penalty_beta,
            'eval_hits_filepath': str(eval_hits_filepath),
            'eval_best_hits_filepath': str(eval_best_hits_filepath),
        })

        # Save results to avoid losing data if something crashes
        summary_file = out_dirpath / "results_final.json"
        with open(summary_file, 'w') as f:
            json.dump(convert_numpy_types(results), f, indent=2)


    # Get result with the best mean_penalized_score
    best_name, best_result = max(results.items(), key=lambda x: x[1].get('mean_penalized_score', 0))
    logger.info(f"GWM with best MRPOS: {best_name} (Overlap: {best_result['mean_overlap_score']:.4f}, MRPOS: {best_result['mean_overlap_score']:.4f}, GWMs: {best_result['n_gwms']:,})")

    # Generate report
    report_filepath = out_dirpath / "evaluation_report.txt"
    generate_evaluation_report(results, report_filepath)
    logger.info(f"Wrote evaluation report to {report_filepath}")

    # Generate comparison summary plot
    plots_filepath = out_dirpath / "evaluation_plots.png"
    plot_evaluation_summary(results, plots_filepath)
    logger.info(f"Wrote evaluation plots to {plots_filepath}")

    best_result["name"] = best_name
    return best_result


def create_and_evaluate_motif_gwms(
    stat_matches_filepath,
    top_matches_filepath,
    clusters_filepath,
    annotated_subclusters_filepath,
    reference_clusters_filepath,
    overlap_penalty_alpha,
    overlap_penalty_beta,
    out_dirpath,
    n_jobs,
):

    configs = [
        {'method': 'hdbscan', 'min_cluster_size': 10, 'use_svd': True, 'target_variance': 0.90,
        'cluster_selection_method': 'eom', 'cluster_selection_epsilon': 0.1, 'name': 'hdb_svd90_mcs10_eps01'},
        {'method': 'hdbscan', 'min_cluster_size': 10, 'use_svd': True, 'target_variance': 0.95,
        'cluster_selection_method': 'eom', 'cluster_selection_epsilon': 0.1, 'name': 'hdb_svd95_mcs10_eps01'},
        {'method': 'hdbscan', 'min_cluster_size': 10, 'use_svd': False,
        'cluster_selection_method': 'eom', 'cluster_selection_epsilon': 0.1, 'name': 'hdb_mcs10_eps01'}
    ]

    min_matches = (20,)
    min_core_genes = (2,)
    core_threshold = (0.8, 0.9,)
    min_gene_prob = (0.2,)
    gwm_hyperparams = list(product(
        min_matches,
        min_core_genes,
        core_threshold,
        min_gene_prob,
    ))

    clustering_results = generate_subcluster_motifs(
        configs,
        stat_matches_filepath,
        top_matches_filepath,
        out_dirpath,
        n_jobs
        )

    gwm_results = generate_subcluster_gwms(
        clustering_results,
        gwm_hyperparams,
        clusters_filepath,
        out_dirpath,
    )

    best_result = evaluate_subcluster_gwms(
        gwm_results,
        annotated_subclusters_filepath,
        reference_clusters_filepath,
        overlap_penalty_alpha,
        overlap_penalty_beta,
        out_dirpath
    )

    return best_result


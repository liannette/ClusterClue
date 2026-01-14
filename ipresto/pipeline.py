import time
import logging
from pathlib import Path
from ipresto.clusters.tokenize.orchestrator import TokenizeOrchestrator
from ipresto.clusters.orchestrator import PreprocessOrchestrator
from ipresto.presto_stat.orchestrator import StatOrchestrator
from ipresto.presto_top.orchestrator import TopOrchestrator
from ipresto.gwms.create_motifs import generate_subcluster_motifs
from ipresto.gwms.evaluate_motifs import select_best_motif_set


logger = logging.getLogger(__name__)


def create_new_motifs(
    out_dir_path,
    gbks_dir_path,
    existing_clusterfile,
    exclude_name,
    include_contig_edge_clusters,
    hmm_file_path,
    max_domain_overlap,
    min_genes_per_bgc,
    domain_filtering,
    similarity_filtering,
    similarity_cutoff,
    remove_infrequent_genes,
    min_gene_occurrence,
    run_stat,
    stat_modules_file_path,
    stat_pval_cutoff,
    stat_n_families_range,
    run_top,
    top_model_file_path,
    top_n_topics, 
    top_amplify, 
    top_iterations,
    top_chunksize, 
    top_update, 
    top_visualise,
    top_alpha,
    top_beta,
    top_plot,
    top_feat_num,
    top_min_feat_score,
    k_values,
    reference_subclusters_filepath,
    reference_gbks_dirpath,
    cores,
    verbose,
    log_queue
):
    """
    Runs the entire pipeline.
    """
    start_time = time.time()

    out_dir_path = Path(out_dir_path)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Preprocessing clusters
    logger.info("=== Preprocessing clusters ===")
    preprocess_dir_path = out_dir_path / "preprocess"
    clusters_file_path = PreprocessOrchestrator().run(
        preprocess_dir_path,
        existing_clusterfile,
        gbks_dir_path,
        hmm_file_path,
        exclude_name,
        include_contig_edge_clusters,
        min_genes_per_bgc,
        max_domain_overlap,
        domain_filtering,
        similarity_filtering,
        similarity_cutoff,
        remove_infrequent_genes,
        min_gene_occurrence,
        cores,
        verbose,
        log_queue,
    )

    # Step 2: Statistical subcluster detection (PRESTO-STAT)
    stat_dir_path = out_dir_path / "stat_subclusters"
    if run_stat:
        logger.info("=== PRESTO-STAT: statistical subcluster detection ===")
        StatOrchestrator().run(
            stat_dir_path,
            clusters_file_path,
            stat_modules_file_path,
            stat_pval_cutoff,
            stat_n_families_range,
            min_genes_per_bgc,
            cores,
            verbose,
        )

    # Step 3: Topic modelling for subcluster motif detection (PRESTO-TOP)
    top_dir_path = out_dir_path / "top_subclusters"
    if run_top:
        logger.info("=== PRESTO-TOP: subcluster detection using topic modelling ===")
        TopOrchestrator().run(
            top_dir_path,
            clusters_file_path,
            top_model_file_path,
            top_n_topics, 
            top_amplify, 
            top_iterations,
            top_chunksize, 
            top_update, 
            top_visualise,
            top_alpha,
            top_beta,
            top_plot,
            top_feat_num,
            top_min_feat_score,
            cores,
            verbose,
        )

    # Step 4: Create GWMs
    logger.info("=== Create Subcluster Motif gene weight matrices (GWMs) ===")
    stat_matches_filepath = stat_dir_path / "detected_stat_modules.txt"
    top_matches_filepath = top_dir_path / "matches_per_topic_filtered.txt"

    gwm_dirpath = out_dir_path / "motif_gwms"
    motifs_filepaths = generate_subcluster_motifs(
        clusters_file_path,
        stat_matches_filepath,
        top_matches_filepath,
        k_values,
        gwm_dirpath
    )

    # select best motifs based on evaluation (to be implemented)
    evaluation_dirpath = gwm_dirpath / "evaluation"
    best_motif_filepath = select_best_motif_set(
        motifs_filepaths=motifs_filepaths,
        output_dirpath=evaluation_dirpath,
        reference_subclusters_filepath=reference_subclusters_filepath,
        reference_gbks_dirpath=reference_gbks_dirpath,
        hmm_file_path=hmm_file_path,
        max_domain_overlap=max_domain_overlap,
        cores=cores,
        verbose=verbose,
        log_queue=log_queue,
    )

    # Write best motifs to final output file
    final_motif_filepath = out_dir_path / "subcluster_motifs.txt"
    best_motifs = best_motif_filepath.read_text()
    final_motif_filepath.write_text(best_motifs)
    logger.info(f"Wrote best motif set to {final_motif_filepath}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info("Total runtime: %d hours and %d minutes", int(hours), int(minutes))


def detect_existing_motifs(
    out_dir_path,
    gbks_dir_path,
    existing_clusterfile,
    exclude_name,
    include_contig_edge_clusters,
    hmm_file_path,
    max_domain_overlap,
    min_genes_per_bgc,
    domain_filtering,
    similarity_filtering,
    similarity_cutoff,
    remove_infrequent_genes,
    min_gene_occurrence,
    run_stat,
    stat_modules_file_path,
    stat_pval_cutoff,
    stat_n_families_range,
    run_top,
    top_model_file_path,
    top_n_topics, 
    top_amplify, 
    top_iterations,
    top_chunksize, 
    top_update, 
    top_visualise,
    top_alpha,
    top_beta,
    top_plot,
    top_feat_num,
    top_min_feat_score,
    k_values,
    reference_subclusters_filepath,
    reference_gbks_dirpath,
    cores,
    verbose,
    log_queue
    ):

    start_time = time.time()

    out_dir_path = Path(out_dir_path)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Preprocessing clusters
    logger.info("=== Preprocessing input biosynthetic gene clusters ===")
    preprocess_dir_path = out_dir_path / "preprocess"
    clusters_file_path = preprocess_dir_path / "clusters_all_domains.csv"
    gene_counts_file_path = preprocess_dir_path / "clusters_all_domains_gene_counts.txt"
    if clusters_file_path.is_file():
        logger.info(f"Skipping tokenisation step, because the file already exists: {clusters_file_path}")
    elif existing_clusterfile:
        logger.info(f"Using provided file of tokenized BGCs: {existing_clusterfile}.")
        clusters_file_path.write_text(Path(existing_clusterfile).read_text())
    else:
        TokenizeOrchestrator().run(
            clusters_file_path,
            gene_counts_file_path,
            gbks_dir_path,
            hmm_file_path,
            exclude_name,
            include_contig_edge_clusters,
            min_genes_per_bgc,
            max_domain_overlap,
            cores,
            verbose,
            log_queue,
        )

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info("Total runtime: %d hours and %d minutes", int(hours), int(minutes))

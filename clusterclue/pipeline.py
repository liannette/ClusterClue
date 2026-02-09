import logging
from pathlib import Path
from clusterclue.clusters.tokenize.orchestrator import TokenizeOrchestrator
from clusterclue.clusters.orchestrator import PreprocessOrchestrator

logger = logging.getLogger(__name__)


def create_new_motifs(
    cmd,
    log_queue
):
    """
    Runs the entire pipeline.
    """
    # import here to avoid importing heavy dependencies (e.g. gensim) when 
    # the user only wants to detect existing motifs
    from clusterclue.presto_stat.orchestrator import StatOrchestrator
    from clusterclue.presto_top.orchestrator import TopOrchestrator
    from clusterclue.gwms.create_motifs import generate_subcluster_motifs
    from clusterclue.gwms.select_best import select_best_motif_set
    from clusterclue.evaluate.presto_predictions import evaluate_presto_stat, evaluate_presto_top

    out_dirpath = Path(cmd.out_dir_path)
    out_dirpath.mkdir(parents=True, exist_ok=True)

    # Step 1: Preprocessing clusters
    logger.info("=== Preprocessing clusters ===")
    preprocess_dirpath = out_dirpath / "preprocess"
    clusters_file_path = PreprocessOrchestrator().run(
        preprocess_dirpath,
        cmd.gbks_dir_path,
        cmd.hmm_file_path,
        cmd.include_contig_edge_clusters,
        cmd.min_genes_per_bgc,
        cmd.max_domain_overlap,
        cmd.similarity_filtering,
        cmd.similarity_cutoff,
        cmd.remove_infrequent_genes,
        cmd.min_gene_occurrence,
        cmd.cores,
        cmd.verbose,
        log_queue,
    )


    # Step 2: Statistical subcluster detection (PRESTO-STAT)
    stat_dirpath = out_dirpath / "presto_stat_subclusters"
    if cmd.run_stat:
        logger.info("=== PRESTO-STAT: statistical subcluster detection ===")
        StatOrchestrator().run(
            stat_dirpath,
            clusters_file_path,
            cmd.stat_modules_file_path,
            cmd.stat_pval_cutoff,
            cmd.stat_n_families_range,
            cmd.min_genes_per_bgc,
            cmd.cores,
            cmd.verbose,
        )

    # Step 3: Topic modelling for subcluster motif detection (PRESTO-TOP)
    top_dirpath = out_dirpath / "presto_top_subclusters"
    if cmd.run_top:
        logger.info("=== PRESTO-TOP: subcluster detection using topic modelling ===")
        TopOrchestrator().run(
            top_dirpath,
            clusters_file_path,
            cmd.top_model_file_path,
            cmd.top_n_topics, 
            cmd.top_amplify, 
            cmd.top_iterations,
            cmd.top_chunksize, 
            cmd.top_update, 
            cmd.top_visualise,
            cmd.top_alpha,
            cmd.top_beta,
            cmd.top_plot,
            cmd.top_feat_num,
            cmd.top_min_feat_score,
            cmd.cores,
        )

    # PRESTO evaluation
    logger.info("=== Evaluating PRESTO subcluster detections against reference subclusters ===")
    
    evaluation_dirpath = out_dirpath / "evaluation" 
    evaluation_dirpath.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Preprocessing BGCs containing annotated subclusters from {cmd.reference_subclusters_filepath}")
    prep_ref_dirpath = evaluation_dirpath / "preprocess_reference_clusters"
    prep_ref_dirpath.mkdir(parents=True, exist_ok=True)

    ref_clusters_filepath = prep_ref_dirpath / "clusters.tsv"
    ref_domain_hits_filepath = prep_ref_dirpath / "all_domain_hits.txt"

    if ref_clusters_filepath.is_file() and ref_domain_hits_filepath.is_file():
        logger.info("Skipping tokenisation step for reference BGCs, because the files already exist.")
    else:
        TokenizeOrchestrator().run(
            clusters_file_path=ref_clusters_filepath,
            gene_counts_file_path=prep_ref_dirpath / "gene_counts.tsv",
            gbks_dir_path=cmd.reference_gbks_dirpath,
            hmm_file_path=cmd.hmm_file_path,
            include_contig_edge_clusters=True,
            min_genes=0,
            max_domain_overlap=cmd.max_domain_overlap,
            cores=cmd.cores,
            verbose=cmd.verbose,
            log_queue=log_queue,
        )
    stat_evaluate_dirpath = evaluation_dirpath / "presto_stat_subclusters"
    if stat_evaluate_dirpath.is_dir() and any(stat_evaluate_dirpath.iterdir()):
        logger.info(f"Skipping PRESTO-STAT evaluation step, because the directory already exists and is not empty: {stat_evaluate_dirpath}")
    else:
        evaluate_presto_stat(
            reference_subclusters_filepath=cmd.reference_subclusters_filepath,
            reference_clusters_filepath=ref_clusters_filepath,
            stat_modules=stat_dirpath / "stat_modules.txt",
            output_dirpath=evaluation_dirpath / "presto_stat_subclusters",
        )
    top_evaluate_dirpath = evaluation_dirpath / "presto_top_subclusters"
    if top_evaluate_dirpath.is_dir() and any(top_evaluate_dirpath.iterdir()):
        logger.info(f"Skipping PRESTO-TOP evaluation step, because the directory already exists and is not empty: {top_evaluate_dirpath}")
    else:
        evaluate_presto_top(
            reference_subclusters_filepath=cmd.reference_subclusters_filepath,
            reference_clusters_filepath=ref_clusters_filepath,
            top_model_file_path=top_dirpath / "lda_model",
            number_of_topics=cmd.top_n_topics,
            output_dirpath=evaluation_dirpath / "presto_top_subclusters",
        )


    # Step 4: Create GWMs
    logger.info("=== Create Subcluster Motif gene weight matrices (GWMs) ===")
    motifs_dirpath = out_dirpath / "clusterclue_motifs"
    if motifs_dirpath.is_dir() and any(motifs_dirpath.iterdir()):
        logger.info(f"Skipping motif generation step, because the directory already exists and is not empty: {motifs_dirpath}")
    else:
        motifs_dirpath.mkdir(parents=True, exist_ok=True)
        stat_matches_filepath = stat_dirpath / "detected_stat_modules.txt"
        top_matches_filepath = top_dirpath / "matches_per_topic_filtered.txt"
        gwms_dirpath = generate_subcluster_motifs(
            clusters_file_path,
            stat_matches_filepath,
            top_matches_filepath,
            cmd.k_values,
            motifs_dirpath,
        )

        select_best_motif_set(
            output_dirpath=motifs_dirpath,
            gwms_dirpath=gwms_dirpath,
            reference_subclusters_filepath=cmd.reference_subclusters_filepath,
            clusters_filepath=ref_clusters_filepath,
        )

        # visualize the best motif set
        if cmd.visualise_evaluation:
            from clusterclue.evaluate.visualize import visualize_evaluation_results
            visualize_evaluation_results(
                motifs_dirpath,
                ref_clusters_filepath,
                Path(cmd.reference_gbks_dirpath),
                Path(cmd.compound_smiles_filepath),
            )


def detect_existing_motifs(
    cmd,
    log_queue,
    ):
    from clusterclue.gwms.detect_motifs import detect_gwms_in_clusters

    out_dirpath = Path(cmd.out_dir_path)
    out_dirpath.mkdir(parents=True, exist_ok=True)

    # Step 1: Preprocessing clusters
    logger.info("=== Preprocessing input biosynthetic gene clusters ===")
    preprocess_dirpath = out_dirpath / "preprocess"

    clusters_file_path = PreprocessOrchestrator().run(
        out_dir_path=preprocess_dirpath,
        gbks_dir_path=cmd.gbk_dir_path,
        hmm_file_path=cmd.hmm_file_path,
        include_contig_edge_clusters=True,
        min_genes=0,
        max_domain_overlap=cmd.max_domain_overlap,
        similarity_filtering_flag=False,
        sim_cutoff=None,
        remove_infrequent_genes_flag=False,
        min_gene_occurrence=None,
        cores=cmd.cores,
        verbose=cmd.verbose,
        log_queue=log_queue,
    )

    # Step 2: Detect existing motifs
    logger.info("=== Detecting existing subcluster motifs ===")
    hits_filepath = out_dirpath / "detected_motifs.tsv"
    detect_gwms_in_clusters(
        clusters_filepath=clusters_file_path,
        motifs_filepath=cmd.gwms_filepath,
        output_filepath=hits_filepath,
    )

    # Step 3: Visualize detected motif hits
    if cmd.visualize_hits:
        from clusterclue.gwms.detect_motifs import visualise_gwm_hits 
        logger.info("=== Visualizing detected motif hits ===")
        visualization_output_dirpath = out_dirpath / "hit_visualizations"
        visualization_output_dirpath.mkdir(parents=True, exist_ok=True)
        visualise_gwm_hits(
            motif_gwms_filepath=cmd.gwms_filepath,
            motif_hits_filepath=hits_filepath,
            genbank_dirpath=cmd.gbk_dir_path,
            domain_hits_filepath=preprocess_dirpath / "all_domain_hits.txt",
            compound_structures_filepath=cmd.compound_smiles_filepath,
            output_dirpath=visualization_output_dirpath,
        )


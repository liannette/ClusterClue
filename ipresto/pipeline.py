from pathlib import Path

from ipresto.preprocess.orchestrator import PreprocessOrchestrator
from ipresto.presto_stat.orchestrator import StatOrchestrator
from ipresto.presto_top.orchestrator import TopOrchestrator


class IprestoPipeline:
    def run(
        self,
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
        cores,
        verbose,
    ):
        """
        Runs the entire pipeline.
        """
        out_dir_path = Path(out_dir_path)
        out_dir_path.mkdir(parents=True, exist_ok=True)

        # Step 1: Preprocessing clusters
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
        )

        # Step 2: Statistical subcluster detection (PRESTO-STAT)
        if run_stat:
            if verbose:
                print("\n=== PRESTO-STAT: statistical subcluster detection ===")
            stat_dir_path = out_dir_path / "stat_subclusters"
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

        # Step 3: Topic modelling for ubcluster motif detection (PRESTO-TOP)
        if run_top:
            if verbose:
                print("\n=== PRESTO-TOP: subcluster motif detection using topic modelling ===")
            top_dir_path = out_dir_path / "top_subclusters"
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

        # if visualize:
        #     if verbose:
        #         print("=== Visualizations ===")
        #     plot_histogram(load_data(preprocessed_data_path), output_path="histogram.png")
        #     plot_comparison(results, output_path="comparison.png")

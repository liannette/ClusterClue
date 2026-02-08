import random
import logging
from pathlib import Path
from multiprocessing import Queue
from clusterclue.clusters.tokenize.orchestrator import TokenizeOrchestrator
from clusterclue.clusters.similarity_filtering import filter_similar_clusters
from clusterclue.clusters.infrequent_genes import remove_infrequent_genes
from typing import Optional

logger = logging.getLogger(__name__)


class PreprocessOrchestrator:
    def run(
        self,
        out_dir_path: str,
        gbks_dir_path: str,
        hmm_file_path: str,
        include_contig_edge_clusters: bool,
        min_genes: int,
        max_domain_overlap: float,
        similarity_filtering_flag: bool,
        sim_cutoff: Optional[float],
        remove_infrequent_genes_flag: bool,
        min_gene_occurrence: Optional[int],
        cores: int,
        verbose: bool,
        log_queue: Queue,
    ) -> str:
        """Orchestrate all preprocessing steps.

        Preprocess BGCs by tokenizing, filtering non-biosynthetic domains, and removing similar clusters.

        Args:
            out_dir_path (str): Output directory path.
            gbks_dir_path (str): Directory path containing GenBank files.
            hmm_file_path (str): Path to HMM file.
            include_contig_edge_clusters (bool): Whether to include clusters that lie on a contig edge.
            min_genes (int): Minimum number of genes.
            max_domain_overlap (float): If two domains overlap more than this value, only the domain with the highest score is kept.
            similarity_filtering (bool): Whether to filter similar clusters.
            sim_cutoff (float): Similarity cutoff value for filtering clusters.
            remove_infrequent_genes_flag (bool): Whether to remove infrequent genes.
            min_gene_occurrence (int): Minimum number of occurrences for a gene to be retained.
            cores (int): Number of cores to use.
            verbose (bool): Whether to print verbose output.
            log_queue (multiprocessing.Queue): Queue for logging in multiprocessing.

        Returns:
            str: Path to the final preprocessed clusters file.
        """
        # Set random seed for reproducibility
        random.seed(595)

        out_dir = Path(out_dir_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Tokenize genes into protein domain combinations
        clusters_file_path = out_dir / "clusters_all_domains.csv"
        gene_counts_file_path = out_dir / "clusters_all_domains_gene_counts.txt"
        if clusters_file_path.is_file():
            logger.info(f"Skipping tokenisation step, because the file already exists: {clusters_file_path}")
        else:
            TokenizeOrchestrator().run(
                clusters_file_path,
                gene_counts_file_path,
                gbks_dir_path,
                hmm_file_path,
                include_contig_edge_clusters,
                min_genes,
                max_domain_overlap,
                cores,
                verbose,
                log_queue,
            )

        # Remove highly similar clusters
        if similarity_filtering_flag:
            out_file_path = out_dir / "clusters_deduplicated.csv"
            if out_file_path.is_file():
                logger.info(f"Skipping similarity filtering, because the file already exists: {out_file_path}")
            else:
                counts_file_path = out_dir / "clusters_deduplicated_gene_counts.txt"
                representatives_file_path = out_dir / "representative_clusters.txt"
                edge_file_path = out_dir / "edges.txt"
                filter_similar_clusters(
                    clusters_file_path,
                    out_file_path,
                    counts_file_path,
                    representatives_file_path,
                    edge_file_path,
                    sim_cutoff,
                    cores,
                    verbose,
                    log_queue,
                )
            clusters_file_path = out_file_path

        # Remove infrequent genes
        if remove_infrequent_genes_flag:
            out_file_path = out_dir / "clusters_gene_filtered.csv"
            if out_file_path.is_file():
                logger.info(f"Skipping infrequent gene removal, because the file already exists: {out_file_path}")
            else:
                counts_file_path = out_dir / "clusters_gene_filtered_gene_counts.txt"
                clusters_file_path = remove_infrequent_genes(
                    clusters_file_path,
                    out_file_path,
                    counts_file_path,
                    min_genes,
                    min_gene_occurrence,
                    verbose,
                )
            clusters_file_path = out_file_path

        return clusters_file_path

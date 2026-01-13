import logging
from collections import Counter
from itertools import product
from pathlib import Path

from ipresto.clusters.utils import read_clusters
from ipresto.gwms.create.combine_matches import combine_presto_matches
from ipresto.gwms.create.cluster_matches import cluster_matches_kmeans
from ipresto.gwms.create.build_gwms import build_motif_gwms, write_motif_gwms


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


def write_gene_background_count(
    gene_counts: Counter,
    n_clusters: int, 
    out_filepath: Path
    ) -> None:
    """Writes the background counts of tokenised genes to a file."""
    with open(out_filepath, "w") as outfile:
        outfile.write(f"#Total_BGCs\t{n_clusters}\n")
        for tokenized_gene in sorted(gene_counts):
            outfile.write(f"{tokenized_gene}\t{gene_counts[tokenized_gene]}\n")


def generate_subcluster_motifs(      
    clusters_filepath: Path,        
    stat_matches_filepath: Path,
    top_matches_filepath: Path,
    k_values: list[int],
    out_dirpath: Path
    ):

    out_dirpath.mkdir(parents=True, exist_ok=True)

    logger.info("Combining subcluster predictions from stat and top method")
    combined_matches_filepath = out_dirpath / "matches.txt"
    combined_matches = combine_presto_matches(
        stat_matches_filepath, 
        top_matches_filepath, 
        combined_matches_filepath
        )

    logger.info(f"Reading clusters from {clusters_filepath}")
    clusters = read_clusters(clusters_filepath)
    n_clusters = len(clusters)
    logger.info(f"Total number of BGCs: {n_clusters}")

    gene_bg_counts = get_gene_background_count(clusters)
    logger.info(f"Calculated gene background counts across all BGCs for {len(gene_bg_counts)} tokenized genes")
    
    bg_counts_filepath = out_dirpath / "genes_background_count.txt"
    write_gene_background_count(gene_bg_counts, n_clusters, bg_counts_filepath)
    logger.info(f"Wrote gene background counts to {bg_counts_filepath}")

    motif_filepaths = []
    
    for k in k_values:
        subout_dirpath = out_dirpath / f"kmeans_{k}"
        subout_dirpath.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Clustering subcluster matches into {k} motifs using k-means")
        subcluster_motifs = cluster_matches_kmeans(combined_matches, k, subout_dirpath)
        
        logger.info("Creating GWMs with different parameter combinations")
        # min_matches = (5, 25, 50)
        # min_core_genes = (1, 2)
        # core_threshold = (0.6, 0.7, 0.8)
        # min_gene_prob = (0.1, 0.3, 0.5)
        min_matches = (5,)
        min_core_genes = (1,)
        core_threshold = (0.6,)
        min_gene_prob = (0.1,)
        hyperparams = product(
            min_matches,
            min_core_genes,
            core_threshold,
            min_gene_prob,
        )
        for mm, mgc, ct, mgp, in hyperparams:
            subcluster_motifs = build_motif_gwms(subcluster_motifs, gene_bg_counts, n_clusters, mm, mgc, ct, mgp, subout_dirpath)

            motif_filepath = subout_dirpath / f"GWMs_mm{mm}_mgc{mgc}_ct{int(ct * 100)}_mgp{int(mgp * 100)}.txt"
            write_motif_gwms(subcluster_motifs, motif_filepath)
            motif_filepaths.append(motif_filepath)

    return motif_filepaths
import logging
from collections import Counter
from itertools import product
from pathlib import Path

from clusterclue.clusters.utils import read_clusters
from clusterclue.gwms.create.combine_matches import combine_presto_matches
from clusterclue.gwms.create.cluster_matches import cluster_matches_kmeans
from clusterclue.gwms.create.build_gwms import build_motif_gwms, write_motif_gwms


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

    combined_matches_filepath = out_dirpath / "matches.txt"
    combined_matches = combine_presto_matches(
        stat_matches_filepath, 
        top_matches_filepath, 
        combined_matches_filepath
    )

    clusters = read_clusters(clusters_filepath)
    n_clusters = len(clusters)
    logger.info(f"Read {n_clusters} tokenized clusters from {clusters_filepath}")

    gene_bg_counts = get_gene_background_count(clusters)
    bg_counts_filepath = out_dirpath / "genes_background_count.txt"
    write_gene_background_count(gene_bg_counts, n_clusters, bg_counts_filepath)
    logger.info(f"Wrote BGCs counts for {len(gene_bg_counts)} unique tokenized genes to {bg_counts_filepath}")

    motif_filepaths = []
    
    for k in k_values:
        subout_dirpath = out_dirpath / f"kmeans_{k}"
        subout_dirpath.mkdir(parents=True, exist_ok=True)
        
        subcluster_motifs = cluster_matches_kmeans(combined_matches, k, subout_dirpath)
        
        # TODO: make these hyperparameters configurable
        min_matches = (5, 10, 20)
        min_core_genes = (1, 2)
        core_threshold = (0.6, 0.7, 0.8)
        min_gene_prob = (0.1, 0.2, 0.3)
        hyperparams = product(
            min_matches,
            min_core_genes,
            core_threshold,
            min_gene_prob,
        )
        for mm, mgc, ct, mgp in hyperparams:
            logger.info(
                f"Building GWMs for k={k}, min_matches={mm}, "
                f"min_core_genes={mgc}, core_threshold={ct}, "
                f"min_gene_prob={mgp}...")
                
            motifs_with_gwms = build_motif_gwms(
                subcluster_motifs, 
                gene_bg_counts, 
                n_clusters, 
                mm, 
                mgc, 
                ct, 
                mgp
                )

            motif_filepath = subout_dirpath / f"GWMs_k{k}_mm{mm}_mgc{mgc}_ct{int(ct * 100)}_mgp{int(mgp * 100)}.txt"
            write_motif_gwms(motifs_with_gwms, motif_filepath)
            motif_filepaths.append(motif_filepath)

    return motif_filepaths
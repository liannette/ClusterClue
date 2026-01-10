import logging
from ipresto.gwms.combine_matches import combine_presto_matches
from ipresto.gwms.cluster_matches import cluster_matches_kmeans
from ipresto.preprocess.utils import read_clusters
from collections import Counter
from itertools import product


logger = logging.getLogger(__name__)


def get_gene_background_count(clusters, out_filepath):
    gene_counts = Counter()
    for tokenized_genes in clusters.values():
        gene_counts.update(set(tokenized_genes))
    # remove genes without biosynthetic domains
    gene_counts.pop(("-",), None) 
    return gene_counts


def write_gene_background_count(gene_counts, n_clusters, out_filepath):
    with open(out_filepath, "w") as outfile:
        outfile.write(f"#Total_BGCs\t{n_clusters}\n")
        for gene in sorted(gene_counts):
            outfile.write(f"{';'.join(gene)}\t{gene_counts[gene]}\n")
    return gene_counts


def build_motif_gwms(      
    clusters_filepath,        
    stat_matches_filepath,
    top_matches_filepath,
    k_values,
    out_dirpath
    ):

    out_dirpath.mkdir(parents=True, exist_ok=True)

    logger.info("Combining subcluster predictions from stat and top method")
    combined_matches_filepath = out_dirpath / "matches.txt"
    combined_matches = combine_presto_matches(
        stat_matches_filepath, 
        top_matches_filepath, 
        combined_matches_filepath
        )

    logger.info(f"Getting gene background counts from clusterfile {clusters_filepath}")
    clusters = read_clusters(clusters_filepath)
    bg_counts_filepath = out_dirpath / "genes_background_count.txt"
    gene_bg_counts = get_gene_background_count(clusters, bg_counts_filepath)
    logger.info(f"Calculated gene background counts for {len(gene_bg_counts)} tokenized genes")
    
    write_gene_background_count(gene_bg_counts, len(clusters), bg_counts_filepath)
    logger.info(f"Wrote gene background counts to {bg_counts_filepath}")

    for k in k_values:
        subout_dirpath = out_dirpath / f"kmeans_{k}"
        subout_dirpath.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Clustering subcluster matches into {k} motifs using k-means")
        label2geneprobs = cluster_matches_kmeans(combined_matches, k, subout_dirpath)
        
        # logger.info("Creating GWMs with different parameter combinations")
        # # min_matches = (5, 25, 50)
        # # min_core_genes = (1, 2)
        # # core_threshold = (0.6, 0.7, 0.8)
        # # min_gene_prob = (0.1, 0.3, 0.5)
        # min_matches = (5,)
        # min_core_genes = (1,)
        # core_threshold = (0.6,)
        # min_gene_prob = (0.1,)
        # hyperparams = product(
        #     min_matches,
        #     min_core_genes,
        #     core_threshold,
        #     min_gene_prob,
        # )
        # for mm, mgc, ct, mgp, in hyperparams:
        #     build_motif_gwms(label2geneprobs, mm, mgc, ct, mgp, subout_dirpath)


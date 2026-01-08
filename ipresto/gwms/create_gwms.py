import logging
from ipresto.gwms.combine_matches import combine_matches
from ipresto.gwms.cluster_matches import cluster_matches_kmeans
from ipresto.preprocess.utils import read_clusters
from collections import Counter


logger = logging.getLogger(__name__)


def get_gene_background_count(clusters, out_filepath):
    gene_counts = Counter()
    for tokenized_genes in clusters.values():
        gene_counts.update(set(tokenized_genes))
    # remove genes without biosynthetic domains
    gene_counts.pop("-", None) 

    with open(out_filepath, "w") as outfile:
        outfile.write(f"#Total_BGCs\t{len(clusters)}\n")
        for gene in sorted(gene_counts):
            outfile.write(f"{gene}\t{gene_counts[gene]}\n")

    return gene_counts


def create_gwms(      
    clusters_filepath,        
    stat_matches_filepath,
    top_matches_filepath,
    k_values,
    out_dirpath
    ):

    out_dirpath.mkdir(parents=True, exist_ok=True)

    logger.info(f"Getting gene background counts from clusterfile {clusters_filepath}")
    clusters = read_clusters(clusters_filepath)
    bg_counts_filepath = out_dirpath / "genes_background_count.txt"
    if not bg_counts_filepath.exists():
        gene_bg_counts = get_gene_background_count(clusters, bg_counts_filepath)

    logger.info("Combining subcluster predictions from stat and top method")
    combined_matches_filepath = out_dirpath / "matches.txt"
    combined_matches = combine_matches(
        stat_matches_filepath, 
        top_matches_filepath, 
        combined_matches_filepath
        )

    logger.info("Clustering similar subcluster predicions using k-means")
    cluster_matches_kmeans(
        combined_matches,
        k_values,
        out_dirpath
    )

    


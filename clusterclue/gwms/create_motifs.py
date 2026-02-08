import logging
from collections import Counter
from itertools import product
from pathlib import Path
from collections import defaultdict

from clusterclue.clusters.utils import read_clusters
from clusterclue.gwms.create.combine_matches import combine_presto_matches
#from clusterclue.gwms.create.cluster_matches import cluster_matches_kmeans, find_optimal_k
from clusterclue.gwms.create.knn_overlapping_clustering import jaccard_overlapping_clustering, invert_communities
from clusterclue.gwms.create.build_gwms import build_motif_gwms, write_motif_gwms
from clusterclue.classes.subcluster_motif import SubclusterMotif

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


def write_matches_per_group(subcluster_motifs, output_filepath):
    with open(output_filepath, "w") as f:
        for motif_id in sorted(subcluster_motifs.keys()):
            motif = subcluster_motifs[motif_id]
            f.write(f"#motif: {motif_id}, n_matches: {motif.n_matches}\n")
            # sort by bgc_id
            for bgc_id in sorted(motif.matches):
                module = sorted(motif.matches[bgc_id])
                f.write(f"{motif_id}\t{bgc_id}\t{','.join(module)}\n")


def write_gene_probabilities(subcluster_motifs, output_filepath):
    with open(output_filepath, "w") as outfile:
        for motif_id in sorted(subcluster_motifs.keys()):
            motif = subcluster_motifs[motif_id]
            outfile.write(f"#motif: {motif_id}, n_matches: {len(motif.matches)}\n")
            outfile.write(f"{'\t'.join(motif.tokenized_genes)}\n")
            outfile.write(f"{'\t'.join([str(round(p, 3)) for p in motif.probabilities])}\n")


# def collapse_grouped_matches(module2labels, module2bgcs):
#     """
#     Collapses matches per label and per BGC
#     """
#     label_bgc_genes = defaultdict(lambda: defaultdict(set))
#     for module, label in module2labels.items():
#         for bgc_id in module2bgcs[module]:
#             label_bgc_genes[label][bgc_id].update(module)
#     return label_bgc_genes

def collapse_grouped_matches(module2labels, module2bgcs):
    """
    Collapses matches per label and per BGC.
    
    Supports overlapping labels (module → list of labels).

    """
    label_bgc_genes = defaultdict(lambda: defaultdict(set))
    
    for module, labels in module2labels.items():
        bgc_ids = module2bgcs[module]
        for bgc_id in bgc_ids:
            for label in labels:  # <-- loop over all community labels
                label_bgc_genes[label][bgc_id].update(module)
    return label_bgc_genes


def generate_subcluster_motifs(     
    clusters_filepath: Path,        
    stat_matches_filepath: Path,
    top_matches_filepath: Path,
    k_values: list[int],
    out_dirpath: Path,
    cores: int,
    log_queue
    ):

    matches_dirpath = out_dirpath / "matches"
    matches_dirpath.mkdir(parents=True, exist_ok=True)

    combined_matches_filepath = matches_dirpath / "combined_ipresto_matches.txt"
    combined_matches = combine_presto_matches(
        stat_matches_filepath, 
        top_matches_filepath, 
        combined_matches_filepath
    )

    clusters = read_clusters(clusters_filepath)
    n_clusters = len(clusters)
    logger.info(f"Read {n_clusters} tokenized clusters from {clusters_filepath}")

    gene_bg_counts = get_gene_background_count(clusters)
    bg_counts_filepath = matches_dirpath / "genes_background_count.txt"
    write_gene_background_count(gene_bg_counts, n_clusters, bg_counts_filepath)
    logger.info(f"Wrote BGCs counts for {len(gene_bg_counts)} unique tokenized genes to {bg_counts_filepath}")

    # create mapping for subcluster module to bgcs
    module2bgcs = defaultdict(list)
    for bgc_id, module in combined_matches:
        module2bgcs[module].append(bgc_id)
    modules = sorted(list(module2bgcs.keys()))
    logger.info(f"Total unique subcluster modules from combined matches: {len(modules)}")

    # find overlapping communities
    logger.info("Clustering subcluster modules into overlapping communities...")
    communities = jaccard_overlapping_clustering(
        modules,
        jaccard_threshold=0.5,
        min_comm_size=5,
        cores=cores,
        log_queue=log_queue,
    )
    logger.info(f"Identified {len(communities)} overlapping communities")

    # pickle communities for inspection
    communities_filepath = Path.cwd() / "module_communities.pkl"
    with open(communities_filepath, "wb") as f:
        import pickle
        pickle.dump(communities, f)
    logger.info(f"Saved module communities to {communities_filepath}")

    num_communities = len(communities)
    padding = len(str(num_communities))
    module2community_ids = invert_communities(communities)

    module2labels = {
        module: [f"SCM{cid+1:0{padding}d}" for cid in cids]
        for module, cids in module2community_ids.items()
    }
    logger.info(f"Assigned community labels to {len(module2labels)} modules out of {len(modules)} total modules (coverage: {len(module2labels) / len(modules):.2%})")

    # collapse matches per BGC and per label
    label_to_bgc_matches = collapse_grouped_matches(module2labels, module2bgcs)
    logger.info("Collapsed modules per subcluster motif label and per BGC")

    # Now, create SubclusterMotif objects per label
    subcluster_motifs = dict()
    for label, bgc_match in label_to_bgc_matches.items():
        motif = SubclusterMotif(
            motif_id=label,
            matches={bgc_id: sorted(genes) for bgc_id, genes in bgc_match.items()}
            )
        motif.calculate_gene_probabilities()
        subcluster_motifs[label] = motif

    # write clustered matches to file
    clustered_matches_filepath = matches_dirpath / "matches_grouped.txt"
    gene_probs_filepath = matches_dirpath / "probabilities.txt"
    write_matches_per_group(subcluster_motifs, clustered_matches_filepath)
    write_gene_probabilities(subcluster_motifs, gene_probs_filepath)
    logger.info(f"Wrote grouped subcluster predictions to {clustered_matches_filepath} and gene probabilities to {gene_probs_filepath}")

    # build GWMs for optimal k
    gwms_dirpath = out_dirpath / "gwms"
    gwms_dirpath.mkdir(parents=True, exist_ok=True)

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
            f"Building GWMs for min_matches={mm}, "
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

        motif_filepath = gwms_dirpath / f"GWMs_mm{mm}_mgc{mgc}_ct{int(ct * 100)}_mgp{int(mgp * 100)}.txt"
        write_motif_gwms(motifs_with_gwms, motif_filepath)

    return gwms_dirpath



# def generate_subcluster_motifs(      
#     clusters_filepath: Path,        
#     stat_matches_filepath: Path,
#     top_matches_filepath: Path,
#     k_values: list[int],
#     out_dirpath: Path
#     ):

#     matches_dirpath = out_dirpath / "matches"
#     matches_dirpath.mkdir(parents=True, exist_ok=True)

#     combined_matches_filepath = matches_dirpath / "combined_ipresto_matches.txt"
#     combined_matches = combine_presto_matches(
#         stat_matches_filepath, 
#         top_matches_filepath, 
#         combined_matches_filepath
#     )

#     clusters = read_clusters(clusters_filepath)
#     n_clusters = len(clusters)
#     logger.info(f"Read {n_clusters} tokenized clusters from {clusters_filepath}")

#     gene_bg_counts = get_gene_background_count(clusters)
#     bg_counts_filepath = matches_dirpath / "genes_background_count.txt"
#     write_gene_background_count(gene_bg_counts, n_clusters, bg_counts_filepath)
#     logger.info(f"Wrote BGCs counts for {len(gene_bg_counts)} unique tokenized genes to {bg_counts_filepath}")

#     # create mapping for subcluster module to bgcs
#     module2bgcs = defaultdict(list)
#     for bgc_id, module in combined_matches:
#         module2bgcs[module].append(bgc_id)
#     modules = list(module2bgcs.keys())

#     # select best k based
#     optimal_k, optimal_labels = find_optimal_k(
#         modules, 
#         k_values, 
#         out_dirpath
#         )
#     padding = len(str(optimal_k)) # Calculate padding based on k
#     module2labels = {m: f"M{label+1:0{padding}d}" for m, label in zip(modules, optimal_labels)}

#     # collapse matches per BGC and per label
#     label_to_bgc_matches = collapse_grouped_matches(module2labels, module2bgcs)

#     # Now, create SubclusterMotif objects per label
#     subcluster_motifs = dict()
#     for label, bgc_match in label_to_bgc_matches.items():
#         motif = SubclusterMotif(
#             motif_id=label,
#             matches={bgc_id: sorted(genes) for bgc_id, genes in bgc_match.items()}
#             )
#         motif.calculate_gene_probabilities()
#         subcluster_motifs[label] = motif


#     # write clustered matches to file
#     clustered_matches_filepath = matches_dirpath / f"matches_k{optimal_k}.txt"
#     gene_probs_filepath = matches_dirpath / f"probabilities_k{optimal_k}.txt"
#     write_matches_per_group(subcluster_motifs, clustered_matches_filepath)
#     write_gene_probabilities(subcluster_motifs, gene_probs_filepath)
#     logger.info(f"Wrote grouped subcluster predictions to {clustered_matches_filepath} and gene probabilities to {gene_probs_filepath}")

#     # build GWMs for optimal k
#     gwms_dirpath = out_dirpath / "gwms"
#     gwms_dirpath.mkdir(parents=True, exist_ok=True)

#     # TODO: make these hyperparameters configurable
#     min_matches = (5, 10, 20)
#     min_core_genes = (1, 2)
#     core_threshold = (0.6, 0.7, 0.8)
#     min_gene_prob = (0.1, 0.2, 0.3)
#     hyperparams = product(
#         min_matches,
#         min_core_genes,
#         core_threshold,
#         min_gene_prob,
#     )
#     for mm, mgc, ct, mgp in hyperparams:
#         logger.info(
#             f"Building GWMs for k={optimal_k}, min_matches={mm}, "
#             f"min_core_genes={mgc}, core_threshold={ct}, "
#             f"min_gene_prob={mgp}...")
            
#         motifs_with_gwms = build_motif_gwms(
#             subcluster_motifs, 
#             gene_bg_counts, 
#             n_clusters, 
#             mm, 
#             mgc, 
#             ct, 
#             mgp
#             )

#         motif_filepath = gwms_dirpath / f"GWMs_k{optimal_k}_mm{mm}_mgc{mgc}_ct{int(ct * 100)}_mgp{int(mgp * 100)}.txt"
#         write_motif_gwms(motifs_with_gwms, motif_filepath)

#     return gwms_dirpath

    
#     # for k in k_values:
        
#         # subcluster_motifs = cluster_matches_kmeans(combined_matches, k, matches_dirpath)
        
#         # # TODO: make these hyperparameters configurable
#         # min_matches = (5, 10, 20)
#         # min_core_genes = (1, 2)
#         # core_threshold = (0.6, 0.7, 0.8)
#         # min_gene_prob = (0.1, 0.2, 0.3)
#         # hyperparams = product(
#         #     min_matches,
#         #     min_core_genes,
#         #     core_threshold,
#         #     min_gene_prob,
#         # )
#         # for mm, mgc, ct, mgp in hyperparams:
#         #     logger.info(
#         #         f"Building GWMs for k={k}, min_matches={mm}, "
#         #         f"min_core_genes={mgc}, core_threshold={ct}, "
#         #         f"min_gene_prob={mgp}...")
                
#         #     motifs_with_gwms = build_motif_gwms(
#         #         subcluster_motifs, 
#         #         gene_bg_counts, 
#         #         n_clusters, 
#         #         mm, 
#         #         mgc, 
#         #         ct, 
#         #         mgp
#         #         )

#         #     motif_filepath = gwms_dirpath / f"GWMs_k{k}_mm{mm}_mgc{mgc}_ct{int(ct * 100)}_mgp{int(mgp * 100)}.txt"
#         #     write_motif_gwms(motifs_with_gwms, motif_filepath)
    
#     # return gwms_dirpath
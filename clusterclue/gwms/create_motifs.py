import logging
from collections import Counter
from itertools import product
from pathlib import Path
from collections import defaultdict
from typing import Dict, DefaultDict

from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import MultiLabelBinarizer


from clusterclue.clusters.utils import read_clusters
from clusterclue.gwms.create.combine_matches import combine_presto_matches
from clusterclue.gwms.create.build_gwms import build_motif_gwms, write_motif_gwms
from clusterclue.classes.subcluster_motif import SubclusterMotif
from clusterclue.gwms.create.kmeans_clustering import run_kmeans_pipeline

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


def collapse_grouped_matches(
    module2labels: Dict[str, str],
    module2bgcs: Dict[str, list[str]]
     ) -> DefaultDict[str, DefaultDict[str, set]]:
    """
    Collapses matches per label and per BGC.

    Assumes that each module is assigned to only one label.
    """
    label_bgc_genes = defaultdict(lambda: defaultdict(set))
    for module, label in module2labels.items():
        for bgc_id in module2bgcs[module]:
            label_bgc_genes[label][bgc_id].update(module)
    return label_bgc_genes


def create_sparse_matrix(modules):
    mlb = MultiLabelBinarizer(sparse_output=True)
    X = mlb.fit_transform(modules)
    return X


def get_auto_batch_size(n_samples):
    """
    Automatically determine batch size for MiniBatchKMeans.
    - Uses ~5% of n_samples
    - Ensures min 500 and max 50000
    """
    batch = max(500, int(n_samples * 0.05))  # 5% of samples
    batch = min(batch, 50000)                # cap at 50k
    return batch


def minibatch_kmeans_clustering(X, k, batch_size):
    kmeans = MiniBatchKMeans(
        n_clusters=int(k),
        batch_size=batch_size,
        init="k-means++",
        n_init=100,
        max_iter=1000,
        tol=1e-5,
        random_state=42, 
        )
    kmeans.fit(X)
    logger.info(f"KMeans converged in {kmeans.n_iter_} iterations (inertia: {kmeans.inertia_:.2f})")
    return kmeans.labels_


def generate_subcluster_motifs(      
    clusters_filepath: Path,        
    stat_matches_filepath: Path,
    top_matches_filepath: Path,
    target_variance_explained: float,
    k_values: list[int],
    out_dirpath: Path,
    min_matches = (10, 20),
    min_core_genes = (2,),
    core_threshold = (0.8, 0.9, 0.95, 1.0),
    min_gene_prob = (0.2, 0.3),
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
    modules = list(module2bgcs.keys())
    logger.info

    # subsample for quick diagnostics
    run_kmeans_pipeline(
        modules,
        k_values=k_values,
        use_svd=True,
        target_variance=target_variance_explained,
        n_init=3,
        n_jobs=1,
        outdir=out_dirpath / "kmeans_subsample",
        subsample_n=50_000,
    )

    all_labels, mlb, results = run_kmeans_pipeline(
        modules,
        k_values        = k_values,
        use_svd         = True,
        target_variance = target_variance_explained,
        n_init          = 3,
        n_jobs          = 1,
        outdir          = out_dirpath / "kmeans",
        subsample_n     = None,
    )
    gwms_dirpath = out_dirpath / "gwms"
    gwms_dirpath.mkdir(parents=True, exist_ok=True)

    for k, labels in all_labels.items():

        padding = len(str(k)) # Calculate padding based on k
        module2label = {m: f"M{label+1:0{padding}d}" for m, label in zip(modules, labels)}

        # collapse matches per BGC and per label
        # if there are multiple subcluster modules with the same label in the same BGC they should be merged into one match
        label_to_bgc_matches = collapse_grouped_matches(module2label, module2bgcs)

        # Now, create SubclusterMotif objects per label
        subcluster_motifs = dict()
        for label, bgc_match in label_to_bgc_matches.items():
            motif = SubclusterMotif(
                motif_id=label,
                matches={bgc_id: sorted(genes) for bgc_id, genes in bgc_match.items()}
                )
            motif.calculate_gene_probabilities()
            subcluster_motifs[label] = motif

        # write clustered matches to file (files are only for manual inspection, can be removed in the future )
        clustered_matches_filepath = matches_dirpath / f"matches_k{k}.txt"
        gene_probs_filepath = matches_dirpath / f"probabilities_k{k}.txt"
        write_matches_per_group(subcluster_motifs, clustered_matches_filepath)
        write_gene_probabilities(subcluster_motifs, gene_probs_filepath)
        logger.info(f"Wrote grouped subcluster predictions to {clustered_matches_filepath} and gene probabilities to {gene_probs_filepath}")

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

            motif_filepath = gwms_dirpath / f"GWMs_k{k}_mm{mm}_mgc{mgc}_ct{int(ct * 100)}_mgp{int(mgp * 100)}.txt"
            write_motif_gwms(motifs_with_gwms, motif_filepath)

    return gwms_dirpath

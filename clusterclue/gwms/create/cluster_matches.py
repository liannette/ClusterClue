import logging
from collections import defaultdict
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter

from clusterclue.gwms.subcluster_motif import SubclusterMotif


logger = logging.getLogger(__name__)


def create_sparse_matrix(modules):
    mlb = MultiLabelBinarizer(sparse_output=True)
    X = mlb.fit_transform(modules)
    return X


def _get_auto_batch_size(n_samples):
    """
    Automatically determine batch size for MiniBatchKMeans.
    - Uses ~5% of n_samples
    - Ensures min 500 and max 50000
    """
    batch = max(500, int(n_samples * 0.05))  # 5% of samples
    batch = min(batch, 50000)                # cap at 50k
    return batch


def kmeans_clustering(X, k, batch_size):
    kmeans = MiniBatchKMeans(
        n_clusters=int(k),
        batch_size=batch_size,
        init="k-means++",
        n_init=10,
        random_state=42, 
        )
    kmeans.fit(X)
    return kmeans.labels_


def collapse_grouped_matches(module2labels, module2bgcs):
    """
    Collapses matches per label and per BGC
    """
    label_bgc_genes = defaultdict(lambda: defaultdict(set))
    for module, label in module2labels.items():
        for bgc_id in module2bgcs[module]:
            label_bgc_genes[label][bgc_id].update(module)
    return label_bgc_genes


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


def cluster_matches_kmeans(matches, k, output_dirpath):

    # create mapping for subcluster module to bgcs
    module2bgcs = defaultdict(list)
    for bgc_id, module in matches:
        module2bgcs[module].append(bgc_id)
    modules = list(module2bgcs.keys())

    # perform k-means clustering
    X = create_sparse_matrix(modules)
    batch_size = _get_auto_batch_size(X.shape[0])
    logger.info(
        f"Clustering {X.shape[0]} unique subcluster modules with {X.shape[1]} "
        f"unique tokenised genes (k={k}, batch_size={batch_size})..."
        )
    labels = kmeans_clustering(X, k, batch_size)

    padding = len(str(k)) # Calculate padding based on k
    module2labels = {m: f"M{label+1:0{padding}d}" for m, label in zip(modules, labels)}

    # collapse matches per BGC and per label
    label_to_bgc_matches = collapse_grouped_matches(module2labels, module2bgcs)

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
    clustered_matches_filepath = output_dirpath / "motif_matches.txt"
    gene_probs_filepath = output_dirpath / "motif_gene_probabilities.txt"
    write_matches_per_group(subcluster_motifs, clustered_matches_filepath)
    write_gene_probabilities(subcluster_motifs, gene_probs_filepath)
    logger.info(f"Wrote clustered subcluster predictions to {clustered_matches_filepath} and gene probabilities to {gene_probs_filepath}")

    return subcluster_motifs
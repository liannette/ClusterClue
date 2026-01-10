import logging
from collections import defaultdict
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter
# from scipy.sparse import csr_matrix

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


def kmeans_clustering(X, k):
    batch_size = _get_auto_batch_size(X.shape[0])
    logger.info(f"Clustering with k={k}, batch_size={batch_size}...")
    kmeans = MiniBatchKMeans(
        n_clusters=int(k),
        batch_size=batch_size,
        init="k-means++",
        n_init=10,
        random_state=42, 
        )
    kmeans.fit(X)
    return kmeans.labels_


def collapse_grouped_matches(module2labels, modules2bgcs):
    collapsed_matches = defaultdict(lambda: defaultdict(set))
    for module, label in module2labels.items():
        for bgc_id in modules2bgcs[module]:
            collapsed_matches[label][bgc_id].update(module)

    label2matches = defaultdict(list)
    for label, sc_match in collapsed_matches.items():
        for bgc_id, module in sc_match.items():
            sc_match = (bgc_id, tuple(sorted(module)))
            label2matches[label].append(sc_match)

    return label2matches


def write_matches_per_group(matches_per_label, output_filepath):
    with open(output_filepath, "w") as f:
        for label in sorted(matches_per_label.keys()):
            matches = matches_per_label[label]
            matches.sort(key=lambda x: x[0]) # sort by bgc_id
            f.write(f"#motif: {label}, n_matches: {len(matches)}\n")
            for bgc_id, module in matches:
                f.write(f"{label}\t{bgc_id}\t{','.join(module)}\n")


def calulate_gene_probabilities(label2matches, min_prob):
    label2geneprobs = dict()
    for label, matches in label2matches.items():
        gene_probs = []
        # count gene occurrences
        gene_counter = Counter()
        for bgc_id, module in matches:
            gene_counter.update(set(module))
        # calculate probabilities
        n_matches = len(matches)
        for gene, count in gene_counter.items():
            prob = count / n_matches
            # filter by min_prob
            if prob >= min_prob:
                gene_probs.append((gene, prob))
        # sort genes by probability
        gene_probs.sort(key=lambda x: x[1], reverse=True)
        label2geneprobs[label] = gene_probs
    return label2geneprobs


def write_gene_probabilities(label2geneprobs, label2matches, output_filepath):
    with open(output_filepath, "w") as outfile:
        for label in sorted(label2geneprobs.keys()):
            n_matches = len(label2matches[label])
            gene_probs = label2geneprobs[label]
            if len(gene_probs) > 0:
                genes, probs = list(zip(*gene_probs))
            else:
                genes, probs = [], []

            outfile.write(f"#motif: {label}, n_matches: {n_matches}\n")
            outfile.write(f"{'\t'.join(genes)}\n")
            outfile.write(f"{'\t'.join([str(round(p, 3)) for p in probs])}\n")


def cluster_matches_kmeans(matches, k, output_dirpath):

    # create mapping for subcluster module to bgcs
    module2bgcs = defaultdict(list)
    for bgc_id, module in matches:
        module2bgcs[module].append(bgc_id)
    modules = list(module2bgcs.keys())
    logger.info(f"Total number of unique subcluster modules: {len(modules)}")

    # create sparse matrix
    X = create_sparse_matrix(modules)
    logger.info(f"Prepared sparse matrix for clustering: {X.shape[0]} rows (modules), {X.shape[1]} features (genes)")

    # perform k-means clustering
    labels = kmeans_clustering(X, k)
    padding = len(str(k)) # Calculate padding based on k
    module2labels = {m: f"M{label+1:0{padding}d}" for m, label in zip(modules, labels)}

    # collapse matches per BGC and per label
    label2matches = collapse_grouped_matches(module2labels, module2bgcs)

    # write clustered matches to file
    clustered_matches_filepath = output_dirpath / "motif_matches.txt"
    write_matches_per_group(label2matches, clustered_matches_filepath)
    logger.info(f"Wrote clustered matches to {clustered_matches_filepath}")

    # calculate the gene probabilities for each group
    label2geneprobs = calulate_gene_probabilities(label2matches, min_prob=0.001)
    
    gene_probs_filepath = output_dirpath / "motif_gene_probabilities.txt"
    write_gene_probabilities(label2geneprobs, label2matches, gene_probs_filepath)
    logger.info(f"Wrote gene probabilities to {gene_probs_filepath}")
    
    return label2geneprobs
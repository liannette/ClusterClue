import logging
from collections import defaultdict
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import MultiLabelBinarizer
# from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)


def collapse_grouped_matches(module2labels, modules2bgcs):
    # collapse modules if they have the same group label and are in same BGC

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


def get_auto_batch_size(n_samples):
    """
    Automatically determine batch size for MiniBatchKMeans.
    - Uses ~5% of n_samples
    - Ensures min 500 and max 50000
    """
    batch = max(500, int(n_samples * 0.05))  # 5% of samples
    batch = min(batch, 50000)                # cap at 50k
    return batch


def write_matches_per_group(matches_per_label, output_filepath):
    with open(output_filepath, "w") as f:
        for label in sorted(matches_per_label.keys()):
            matches = matches_per_label[label]
            matches.sort(key=lambda x: x[0]) # sort by bgc_id
            f.write(f"#motif: M{label}, n_matches: {len(matches)}\n")
            for bgc_id, module in matches:
                f.write(f"M{label}\t{bgc_id}\t{','.join(module)}\n")


def cluster_matches_kmeans(matches, k_values, output_dirpath):

    # create mapping for subcluster module to bgcs
    module2bgcs = defaultdict(list)
    for bgc_id, module in matches:
        module2bgcs[module].append(bgc_id)
    logger.info(f"Total number of unique subcluster modules: {len(module2bgcs)}")

    # group subcluster predictions using kmeans clustering
    modules = list(module2bgcs.keys())
    mlb = MultiLabelBinarizer(sparse_output=True)
    X = mlb.fit_transform(modules)
    logger.info(f"Prepared sparse matrix for clustering: {X.shape[0]} rows (modules), {X.shape[1]} features (genes)")

    for k in k_values:

        clustered_matches_filepath = output_dirpath / f"matches_kmeans_{k}.txt"
        if clustered_matches_filepath.exists():
            logger.info(f"Skipping k={k}, output file exists: {clustered_matches_filepath}")
            continue

        batch_size = get_auto_batch_size(len(modules))
        logger.info(f"Clustering with k={k}, batch_size={batch_size}...")
        kmeans = MiniBatchKMeans(
            n_clusters=int(k),
            batch_size=batch_size,
            init="k-means++",
            n_init=10,
            random_state=42, 
            )
        kmeans.fit(X)
        module2labels = {m: label+1 for m, label in zip(modules, kmeans.labels_)}

        label2matches = collapse_grouped_matches(module2labels, module2bgcs)

        write_matches_per_group(label2matches, clustered_matches_filepath)
        logger.info(f"Wrote clustered matches to {clustered_matches_filepath}")
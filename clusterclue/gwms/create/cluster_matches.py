import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import MultiLabelBinarizer



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




def find_optimal_k(modules, k_values, output_dirpath):
    """
    Automatically determine optimal k using multiple metrics.
    Returns best k based on silhouette score (primary criterion).
    """
    # perform k-means clustering
    X = create_sparse_matrix(modules)
    logger.info(f"Created sparse matrix with shape {X.shape} for {len(modules)} modules.")
    batch_size = _get_auto_batch_size(X.shape[0])

    all_labels = []
    silhouette_scores = []
    ch_scores = []
    db_scores = []

    X_dense = X.astype(np.uint8).toarray() # Convert to dense for metric calculations
    
    for k in k_values:
        logger.info(f"Evaluating k={k} (batch_size={batch_size})...")
        labels = minibatch_kmeans_clustering(X, k, batch_size)
        
        all_labels.append(labels)
        silhouette_scores.append(silhouette_score(X, labels))
        ch_scores.append(calinski_harabasz_score(X_dense, labels))
        db_scores.append(davies_bouldin_score(X_dense, labels))
        
        logger.info(
            f"k={k}: silhouette={silhouette_scores[-1]:.4f}, "
            f"calinski_harabasz={ch_scores[-1]:.2f}, "
            f"davies_bouldin={db_scores[-1]:.4f}"
        )
    
    # Optimal k: highest silhouette score
    optimal_k = k_values[np.argmax(silhouette_scores)]
    
    # Verify consistency with other metrics
    ch_optimal = k_values[np.argmax(ch_scores)]
    db_optimal = k_values[np.argmin(db_scores)]
    
    logger.info(
        f"Metrics consensus: silhouette → k={optimal_k}, "
        f"calinski_harabasz → k={ch_optimal}, "
        f"davies_bouldin → k={db_optimal}"
    )
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(k_values, silhouette_scores, 'o-')
    axes[0].axvline(optimal_k, color='r', linestyle='--')
    axes[0].set_xlabel('k')
    axes[0].set_ylabel('Silhouette Score')
    axes[0].set_title(f'Silhouette (optimal k={optimal_k})')
    
    axes[1].plot(k_values, ch_scores, 'o-')
    axes[1].axvline(ch_optimal, color='r', linestyle='--')
    axes[1].set_xlabel('k')
    axes[1].set_ylabel('Calinski-Harabasz Index')
    axes[1].set_title(f'Calinski-Harabasz (optimal k={ch_optimal})')
    
    axes[2].plot(k_values, db_scores, 'o-')
    axes[2].axvline(db_optimal, color='r', linestyle='--')
    axes[2].set_xlabel('k')
    axes[2].set_ylabel('Davies-Bouldin Index')
    axes[2].set_title(f'Davies-Bouldin (optimal k={db_optimal})')
    
    plt.tight_layout()
    plt.savefig(output_dirpath / 'k_selection_metrics.png')

    optimal_labels = all_labels[k_values.index(optimal_k)]
    return optimal_k, optimal_labels



# def cluster_matches_kmeans(matches, k, output_dirpath):

#     # create mapping for subcluster module to bgcs
#     module2bgcs = defaultdict(list)
#     for bgc_id, module in matches:
#         module2bgcs[module].append(bgc_id)
#     modules = list(module2bgcs.keys())

#     # perform k-means clustering
#     X = create_sparse_matrix(modules)
#     batch_size = _get_auto_batch_size(X.shape[0])
#     logger.info(
#         f"Clustering {X.shape[0]} unique subcluster modules with {X.shape[1]} "
#         f"unique tokenised genes (k={k}, batch_size={batch_size})..."
#         )
#     labels = minibatch_kmeans_clustering(X, k, batch_size)

#     padding = len(str(k)) # Calculate padding based on k
#     module2labels = {m: f"M{label+1:0{padding}d}" for m, label in zip(modules, labels)}

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
#     clustered_matches_filepath = output_dirpath / f"matches_k{k}.txt"
#     gene_probs_filepath = output_dirpath / f"probabilities_k{k}.txt"
#     write_matches_per_group(subcluster_motifs, clustered_matches_filepath)
#     write_gene_probabilities(subcluster_motifs, gene_probs_filepath)
#     logger.info(f"Wrote clustered subcluster predictions to {clustered_matches_filepath} and gene probabilities to {gene_probs_filepath}")

#     return subcluster_motifs
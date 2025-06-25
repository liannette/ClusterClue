from ipresto.presto_stat.utils import (
    tokenized_genes_to_string,
    write_module_families,
)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import numpy as np
from joblib import parallel_backend
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend


def create_binary_matrix(modules: dict):
    # create a binary matrix, where rows are modules and columns are genes
    vectorizer = CountVectorizer(
        lowercase=False,
        binary=True,
        dtype=np.int32,
        token_pattern=r"(?u)[^,]+",  # features/genes are separated by ','
    )

    rownames = [module_id for module_id, _ in sorted(modules.items())]
    corpus = [
        tokenized_genes_to_string(module.tokenised_genes)
        for _, module in sorted(modules.items())
    ]

    # corpus must be list of strings
    corpus = [
        tokenized_genes_to_string(m.tokenised_genes) for m in modules.values()
    ]

    sparse_feature_matrix = vectorizer.fit_transform(corpus)
    colnames = vectorizer.get_feature_names()

    return sparse_feature_matrix, colnames, rownames


def run_kmeans(sparse_feature_matrix, k, cores):
    with parallel_backend("loky", n_jobs=cores):
        k_means = KMeans(
            n_clusters=k,
            n_init=20,
            max_iter=1000,
            random_state=595,
            verbose=0,
            tol=1e-6,
        ).fit(sparse_feature_matrix)
    return k_means


def plot_kmeans_elbow(k_range, inertias, output_path):
    """
    Plots and saves the Elbow curve for KMeans clustering.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), inertias, "o-", color="blue")
    plt.title("Elbow Method for Optimal k")
    plt.xlabel("Number of families (k)")
    plt.ylabel("Inertia (WCSS)")
    plt.grid(True)
    plt.xticks(list(k_range))
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def cluster_stat_modules(
    modules: dict, k_range: list, cores: int, out_dir: str, verbose: bool
):
    out_dir = Path(out_dir)

    if verbose:
        print("Creating binary matrix for clustering.")
    sparse_feature_matrix, colnames, rownames = create_binary_matrix(modules)
    if verbose:
        print(
            f"Binary matrix created with dimensions: {sparse_feature_matrix.shape}."
        )

    inertias = []
    for k in k_range:
        if verbose:
            print(
                f"\nClustering {len(modules)} STAT modules into {k} families via "
                "k-means clustering."
            )

        k_means = run_kmeans(sparse_feature_matrix, k, cores)
        inertias.append(k_means.inertia_)
        if verbose:
            print(f"Clustering completed for k={k}. Inertia: {k_means.inertia_}")

        # write the families to a file
        out_file_path = out_dir / f"stat_module_{k}_families.txt"
        write_module_families(rownames, k_means.labels_, out_file_path)
        if verbose:
            print(f"STAT module families for k={k} saved to: {out_file_path}")
    
    return inertias
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import json
from collections import defaultdict, Counter

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import HDBSCAN as SklearnHDBSCAN

from clusterclue.clusters.utils import read_clusters
from clusterclue.gwms.create.combine_matches import combine_presto_matches
from clusterclue.classes.subcluster_motif import SubclusterMotif
from clusterclue.gwms.create.build_gwms import build_motif_gwms, write_motif_gwms
from clusterclue.gwms.create.merge_motifs import merge_similar_motifs


logger = logging.getLogger(__name__)


class ClusteringComparison:
    """Compare different clustering approaches for GWM generation."""
    
    def __init__(self, modules: List[str], module2bgcs: Dict[str, List[str]]):
        self.modules = modules
        self.module2bgcs = module2bgcs
        self.mlb = MultiLabelBinarizer(sparse_output=True)
        self.X = self.mlb.fit_transform(modules)

        self._svd_cache = {}
        
        logger.info(f"Initialized with {len(modules)} modules")
        logger.info(f"Matrix shape: {self.X.shape}")
        logger.info(f"Sparsity: {1 - self.X.nnz / (self.X.shape[0] * self.X.shape[1]):.4f}")
        
    def compute_variance_curve(self, max_components=3000, random_state=42):
        """
        Compute SVD variance saturation curve.
        
        Returns:
            n_components: array of component counts
            cumulative_variance: array of cumulative variance explained
        """
        max_components = min(max_components, self.X.shape[1] - 1)
        
        logger.info(f"Computing SVD variance curve up to {max_components} components")
        
        # Fit SVD with max components
        svd = TruncatedSVD(n_components=max_components, random_state=random_state)
        svd.fit(self.X)
        
        # Get cumulative variance
        cumulative_variance = np.cumsum(svd.explained_variance_ratio_)
        n_components = np.arange(1, max_components + 1)
        
        logger.info(f"Max variance explained: {cumulative_variance[-1]:.4f}")
        
        return n_components, cumulative_variance

    def apply_svd(self, target_variance=0.50, step=50, max_components=3000, random_state=42):
        """Apply SVD dimensionality reduction."""
        # Check cache first
        if target_variance in self._svd_cache:
            logger.info(f"Using cached SVD for target variance {target_variance}")
            return self._svd_cache[target_variance]

        logger.info(f"Computing SVD with target variance {target_variance} (will be cached)")
        
        max_components = min(max_components, self.X.shape[1] - 1)
        step           = min(step, max_components)
        best_n         = None
        cumvar         = 0.0
        n              = step

        # # Determine number of components for target variance
        # svd_scan = TruncatedSVD(n_components=min(100, self.X.shape[1] - 1))
        # svd_scan.fit(self.X)
        # cumvar = np.cumsum(svd_scan.explained_variance_ratio_)
        # n_components = int(np.searchsorted(cumvar, target_variance) + 1)
        while n <= max_components:
            svd    = TruncatedSVD(n_components=n, random_state=random_state)
            svd.fit(self.X)
            cumvar = svd.explained_variance_ratio_.sum()
            logger.info(f"[SVD]   n_components={n:>5}: {cumvar*100:.2f}% variance explained")

            if cumvar >= target_variance:
                cumvar_per_component = svd.explained_variance_ratio_.cumsum()
                best_n   = int((cumvar_per_component >= target_variance).argmax()) + 1
                achieved = cumvar_per_component[best_n - 1] * 100
                logger.info(
                    f"[SVD] Target reached — refined to {best_n} components "
                    f"({achieved:.2f}% variance)"
                )
                break
            n += step

        if best_n is None:
            best_n = min(n - step, max_components)
            logger.warning(
                f"[SVD] Could not reach {target_variance*100:.0f}% within "
                f"{max_components} components. Best: {cumvar*100:.2f}% at "
                f"n_components={best_n}."
            )

        # Apply SVD with determined components
        svd = TruncatedSVD(n_components=best_n, random_state=42)
        X_reduced = svd.fit_transform(self.X).astype(np.float32)
        X_reduced = normalize(X_reduced, norm='l2')  # L2 normalization
        
        variance_explained = svd.explained_variance_ratio_.sum()
        logger.info(f"SVD: {self.X.shape[1]} -> {best_n} dims, "
                   f"variance explained: {variance_explained:.4f}")

        # Cache the result
        result = (X_reduced, variance_explained, best_n)
        self._svd_cache[target_variance] = result
        
        return result


    def clear_svd_cache(self):
        """Clear the SVD cache to free memory if needed."""
        self._svd_cache.clear()
        logger.info("SVD cache cleared")
    
    
    def run_kmeans(self, k: int, use_svd: bool = False, 
                   target_variance: float = 0.95) -> Tuple[np.ndarray, Dict]:
        """Run K-means clustering."""
        logger.info(f"Running K-means with k={k}, use_svd={use_svd}")
        
        if use_svd:
            X_input, var_explained, n_components = self.apply_svd(target_variance)
            n_init = 5
            metadata = {
                'use_svd': True,
                'target_variance': target_variance,
                'variance_explained': var_explained,
                'n_components': n_components
            }
        else:
            n_init = 100
            X_input = self.X
            metadata = {'use_svd': False}

        def get_auto_batch_size(n_samples):
            """Uses ~5% of n_samples, capped between 500 and 50,000."""
            batch = max(500, int(n_samples * 0.05))
            batch = min(batch, 50_000)
            return batch
        
        batch_size = get_auto_batch_size(X_input.shape[0])
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            n_init=n_init,
            init='k-means++',
            random_state=42,
            batch_size=batch_size,
            verbose=0
        )
        labels = kmeans.fit_predict(X_input)

        logger.info(
            f"K-means completed: inertia={kmeans.inertia_:.2f}, "
            f"n_iter={kmeans.n_iter_}"
            )
        
        metadata.update({
            'method': 'kmeans',
            'k': k,
            'n_clusters': len(np.unique(labels)),
            'n_iterations': kmeans.n_iter_,
            'inertia': kmeans.inertia_,
        })
        
        return labels, metadata
    
    def run_hdbscan(self, min_cluster_size: int = 20, 
                    use_svd: bool = True,
                    target_variance: float = 0.95,
                    min_samples: int = None,
                    cluster_selection_epsilon: float = 0.0,
                    n_jobs: int = -1) -> Tuple[np.ndarray, Dict]:
        """Run HDBSCAN clustering."""
        logger.info(f"Running HDBSCAN with min_cluster_size={min_cluster_size}, "
                   f"use_svd={use_svd}")
        
        if use_svd:
            X_input, var_explained, n_components = self.apply_svd(target_variance)
            metadata = {
                'use_svd': True,
                'target_variance': target_variance,
                'variance_explained': var_explained,
                'n_components': n_components
            }
        else:
            logger.info(f"Converting sparse matrix to dense ({self.X.shape})")
            X_input = self.X.toarray().astype(np.float32)
            metadata = {'use_svd': False}
        
        if min_samples is None:
            min_samples = min_cluster_size // 2
        
        clusterer = SklearnHDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method="eom",
            cluster_selection_epsilon=cluster_selection_epsilon,
            n_jobs=n_jobs,
            algorithm='ball_tree',
            leaf_size=40,
            copy=True,
        )
        
        labels = clusterer.fit_predict(X_input)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        
        metadata.update({
            'method': 'hdbscan',
            'min_cluster_size': min_cluster_size,
            'min_samples': min_samples,
            'cluster_selection_epsilon': cluster_selection_epsilon,
            'n_clusters': n_clusters,
            'n_noise_points': int(n_noise),
            'noise_fraction': n_noise / len(labels)
        })
        
        logger.info(f"HDBSCAN found {n_clusters} clusters, "
                   f"{n_noise} noise points ({n_noise/len(labels):.2%})")
        
        return labels, metadata
    
    def labels_to_motifs(self, labels: np.ndarray) -> Dict[str, SubclusterMotif]:
        """Convert cluster labels to SubclusterMotif objects."""
        # Group modules by label
        label_to_modules = defaultdict(list)
        for module, label in zip(self.modules, labels):
            if label != -1:  # Skip noise points from HDBSCAN
                label_to_modules[label].append(module)
        
        # Create BGC-level matches per label
        subcluster_motifs = {}
        for label, modules_list in label_to_modules.items():
            # Collect all BGC matches for this label
            bgc_matches = defaultdict(set)
            for module in modules_list:
                for bgc_id in self.module2bgcs[module]:
                    bgc_matches[bgc_id].update(module)
            
            # Create motif
            motif_id = f"M{label+1:04d}"
            motif = SubclusterMotif(
                motif_id=motif_id,
                matches={bgc_id: sorted(genes) for bgc_id, genes in bgc_matches.items()}
            )
            motif.calculate_gene_probabilities()
            subcluster_motifs[motif_id] = motif
        
        return subcluster_motifs
    
    def calculate_cluster_metrics(self, labels: np.ndarray) -> Dict:
        """Calculate clustering quality metrics."""
        from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
        
        # Filter out noise points for metrics
        mask = labels != -1
        if mask.sum() < 2:
            return {
                'silhouette': None,
                'davies_bouldin': None,
                'calinski_harabasz': None,
                'n_samples_for_metrics': int(mask.sum())
            }
        
        labels_filtered = labels[mask]
        X_filtered = self.X[mask]
        
        # Only calculate if we have multiple clusters
        n_clusters = len(np.unique(labels_filtered))
        if n_clusters < 2:
            return {
                'silhouette': None,
                'davies_bouldin': None,
                'calinski_harabasz': None,
                'n_clusters': n_clusters,
                'n_samples_for_metrics': int(mask.sum())
            }
        
        # Subsample if too large
        if X_filtered.shape[0] > 50000:
            logger.info("Subsampling for metric calculation...")
            indices = np.random.choice(X_filtered.shape[0], 50000, replace=False)
            X_sample = X_filtered[indices]
            labels_sample = labels_filtered[indices]
        else:
            X_sample = X_filtered
            labels_sample = labels_filtered
        
        metrics = {
            'silhouette': silhouette_score(X_sample, labels_sample, metric='euclidean', sample_size=10000),
            'davies_bouldin': davies_bouldin_score(X_sample.toarray(), labels_sample),
            'calinski_harabasz': calinski_harabasz_score(X_sample.toarray(), labels_sample),
            'n_clusters': n_clusters,
            'n_samples_for_metrics': int(mask.sum())
        }
        
        logger.info(f"Cluster metrics: silhouette={metrics['silhouette']:.4f}, "
                   f"davies_bouldin={metrics['davies_bouldin']:.4f}, "
                   f"calinski_harabasz={metrics['calinski_harabasz']:.2f}")
        
        return metrics
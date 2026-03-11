import logging 
from clusterclue.classes.subcluster_motif import SubclusterMotif


logger = logging.getLogger(__name__)


def calculate_motif_similarity(motif1, motif2, threshold=0.1):
    """
    Calculates Jaccard similarity between two motifs based on genes above threshold.
    
    Args:
        motif1, motif2: SubclusterMotif objects
        threshold: minimum probability to consider a gene
    
    Returns:
        float: Jaccard similarity (0 to 1)
    """
    def get_gene_set(motif, prob_threshold):
        """Extract genes with probability >= threshold"""
        if motif.probabilities is None:
            return set()
        return set(
            gene for gene, prob in zip(motif.tokenized_genes, motif.probabilities)
            if prob >= prob_threshold
        )
    
    genes1 = get_gene_set(motif1, threshold)
    genes2 = get_gene_set(motif2, threshold)
    
    if not genes1 or not genes2:
        return 0.0
    
    intersection = len(genes1 & genes2)
    union = len(genes1 | genes2)
    
    return intersection / union if union > 0 else 0.0


def calculate_weighted_similarity(motif1, motif2, threshold=0.1):
    """
    Calculates weighted similarity between two motifs considering probabilities.
    
    Args:
        motif1, motif2: SubclusterMotif objects
        threshold: minimum probability to consider a gene
    
    Returns:
        float: weighted similarity score (0 to 1)
    """
    def get_gene_probs(motif, prob_threshold):
        """Extract genes and probabilities above threshold"""
        if motif.probabilities is None:
            return {}
        return {
            gene: prob 
            for gene, prob in zip(motif.tokenized_genes, motif.probabilities)
            if prob >= prob_threshold
        }
    
    probs1 = get_gene_probs(motif1, threshold)
    probs2 = get_gene_probs(motif2, threshold)
    
    if not probs1 or not probs2:
        return 0.0
    
    # Calculate weighted overlap
    all_genes = set(probs1.keys()) | set(probs2.keys())
    
    numerator = sum(
        min(probs1.get(gene, 0), probs2.get(gene, 0))
        for gene in all_genes
    )
    denominator = sum(
        max(probs1.get(gene, 0), probs2.get(gene, 0))
        for gene in all_genes
    )
    
    return numerator / denominator if denominator > 0 else 0.0


def merge_similar_motifs(
    subcluster_motifs, 
    similarity_threshold=0.8,
    gene_prob_threshold=0.1,
    similarity_metric='weighted'
):
    """
    Merges SubclusterMotif objects that are similar based on their gene sets.
    
    Args:
        subcluster_motifs: dict mapping motif_id to SubclusterMotif objects
        similarity_threshold: minimum similarity to merge motifs (0 to 1)
        gene_prob_threshold: minimum gene probability to include in similarity calc
        similarity_metric: 'jaccard' or 'weighted'
    
    Returns:
        dict: merged motifs with new motif IDs
    """
    import numpy as np
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    
    motif_ids = list(subcluster_motifs.keys())
    n_motifs = len(motif_ids)
    
    if n_motifs == 0:
        return {}
    
    if n_motifs == 1:
        return subcluster_motifs
    
    logger.info(f"Calculating pairwise similarities for {n_motifs} motifs...")
    
    # Calculate pairwise similarity matrix
    similarity_func = (calculate_weighted_similarity 
                      if similarity_metric == 'weighted' 
                      else calculate_motif_similarity)
    
    similarity_matrix = np.zeros((n_motifs, n_motifs))
    for i in range(n_motifs):
        for j in range(i + 1, n_motifs):
            sim = similarity_func(
                subcluster_motifs[motif_ids[i]], 
                subcluster_motifs[motif_ids[j]], 
                gene_prob_threshold
            )
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim
    
    # Convert similarity to distance
    distance_matrix = 1 - similarity_matrix
    
    # Perform hierarchical clustering
    condensed_dist = squareform(distance_matrix, checks=False)
    linkage_matrix = linkage(condensed_dist, method='average')
    
    # Cut tree at similarity threshold
    cluster_labels = fcluster(
        linkage_matrix, 
        t=1 - similarity_threshold, 
        criterion='distance'
    )
    
    # Group motifs by cluster
    from collections import defaultdict
    clusters = defaultdict(list)
    for motif_id, cluster_id in zip(motif_ids, cluster_labels):
        clusters[cluster_id].append(motif_id)
    
    logger.info(f"Found {len(clusters)} clusters from {n_motifs} motifs")
    
    # Merge motifs within each cluster
    merged_motifs = {}
    for cluster_id, cluster_motif_ids in clusters.items():
        if len(cluster_motif_ids) == 1:
            # No merging needed
            motif_id = cluster_motif_ids[0]
            merged_motifs[motif_id] = subcluster_motifs[motif_id]
        else:
            # Merge multiple motifs
            motif_group = [subcluster_motifs[mid] for mid in cluster_motif_ids]
            merged_motif = _merge_motif_group(motif_group, cluster_id)
            merged_motifs[merged_motif.motif_id] = merged_motif
            logger.info(
                f"Merged {len(cluster_motif_ids)} motifs into {merged_motif.motif_id}: "
                f"{', '.join(cluster_motif_ids)}"
            )
    
    return merged_motifs


def _merge_motif_group(motif_group, group_id):
    """
    Merges a group of SubclusterMotif objects into a single motif.
    Properly handles cases where the same BGC appears in multiple motifs.
    
    Args:
        motif_group: list of SubclusterMotif objects to merge
        group_id: identifier for the merged group
    
    Returns:
        SubclusterMotif: merged motif
    """
    from collections import defaultdict
    
    # Collect all genes per BGC across all motifs
    bgc_to_genes = defaultdict(set)
    
    for motif in motif_group:
        if motif.matches:
            for bgc_id, genes in motif.matches.items():
                bgc_to_genes[bgc_id].update(genes)
    
    # Convert sets back to sorted lists
    combined_matches = {
        bgc_id: sorted(genes) 
        for bgc_id, genes in bgc_to_genes.items()
    }
    
    # Create merged motif ID from constituent motifs
    constituent_ids = sorted([m.motif_id for m in motif_group])
    merged_id = f"MG{group_id:03d}_{'_'.join(constituent_ids)}"
    
    # Create new merged motif
    merged_motif = SubclusterMotif(
        motif_id=merged_id,
        matches=combined_matches
    )
    
    # Recalculate gene probabilities based on combined matches
    merged_motif.calculate_gene_probabilities()
    
    return merged_motif

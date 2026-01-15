import logging
import numpy as np


logger = logging.getLogger(__name__)


def passes_min_matches(n_matches, min_matches):
    return True if int(n_matches) >= int(min_matches) else False


def passes_min_core_genes(probabilities, min_core_genes, prob_threshold):
    core_gene_prob = [prob for prob in probabilities if prob >= prob_threshold]
    return True if len(core_gene_prob) >= min_core_genes else False


def remove_low_probabilities(motif, min_probability):
    """
    Removes genes that have a probability lower than min_probability
    """
    filtered_genes = []
    filtered_probabilities = []
    for gene, prob in zip(motif.tokenized_genes, motif.probabilities):
        if prob >= min_probability:
            filtered_genes.append(gene)
            filtered_probabilities.append(prob)
    motif.tokenized_genes = filtered_genes
    motif.probabilities = filtered_probabilities
    return motif


def create_probability_matrix(probabilities):
    """Creates a propability matrix of the tokenised genes"""
    prob_matrix = []
    for prob in probabilities:
        prob_matrix.append([prob, 1-prob])
    return np.array(prob_matrix)


def calculate_weight_matrix(probabilities, background_probabilities):
    prob_matrix = create_probability_matrix(probabilities)
    bg_prob_matrix = create_probability_matrix(background_probabilities)
    # pseudocount = 1e-10
    # weight_matrix = np.log2((prob_matrix + pseudocount) / (bg_prob_matrix + pseudocount))
    with np.errstate(divide='ignore'): # ignore log2(0) warnings
        weight_matrix = np.log2(prob_matrix / bg_prob_matrix)
    return weight_matrix
    

def build_motif_gwms(
    subcluster_motifs,
    background_counts,
    n_clusters,
    min_matches,
    min_core_genes,
    core_threshold,
    min_gene_prob,
    ):
    n_low_matches = 0
    n_low_core_genes = 0
    n_low_genes = 0

    filtered_motifs = dict()
    for motif in subcluster_motifs.values():
        # remove probabilities below the min probability threshold
        motif._filter_low_probabilities(min_gene_prob)

        # check if motif passes filter
        if len(motif.tokenized_genes) < 2:
            n_low_genes += 1
            continue
        if not passes_min_core_genes(motif.probabilities, min_core_genes, core_threshold):
            n_low_core_genes += 1
            continue
        if not passes_min_matches(motif.n_matches, min_matches):
            n_low_matches += 1
            continue

        motif.generate_gwm_with_threshold(background_counts, n_clusters)

        filtered_motifs[motif.motif_id] = motif

    n_initial = len(subcluster_motifs)
    n_filtered = len(filtered_motifs)
    p_filtered = (n_filtered / n_initial) * 100
    p_low_matches = (n_low_matches / n_initial) * 100
    p_low_core_genes = (n_low_core_genes / n_initial) * 100
    p_low_genes = (n_low_genes / n_initial) * 100
    

    logger.info(
        f"Build {n_filtered} ({p_filtered:.2f}%) gene weight matrices for {n_initial} motifs. " 
        f"Removed {n_low_genes} ({p_low_genes:.2f}%) due to low gene number, "
        f"{n_low_core_genes} ({p_low_core_genes:.2f}%) due to low core genes, "
        f"and {n_low_matches} ({p_low_matches:.2f}%) due to low matches."
        )
    return filtered_motifs


def write_motif_gwms(motifs, out_filepath):
    with open(out_filepath, "w") as outfile:
        for motif_id in sorted(motifs):
            motif = motifs[motif_id]
            outfile.write(motif.gwm_to_txt())
    logger.info(f"Wrote motif gene weight matrices to {out_filepath}")

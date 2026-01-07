import logging
from collections import Counter, OrderedDict
from ipresto.preprocess.utils import (
    read_clusters_and_remove_empty,
    write_gene_counts,
    write_clusters,
    count_gene_occurrences,
    count_non_empty_genes,
)

logger = logging.getLogger(__name__)


def remove_infrequent_genes(
    in_file_path: str, 
    out_file_path: str,
    counts_file_path: str,
    min_genes: int, 
    min_gene_occurrence: int, 
    verbose: bool,
) -> str:
    """
    Remove infrequent genes from the clusters.

    Replaces tokenised genes with - if they occur < min_gene_occurrence
    and removes bgcs if they have less than min_genes non-empty genes

    in_file_path: str, path to the input file containing tokenised clusters
    min_genes: int, minimal non-empty genes (domain combinations) a bgc must have to be included
    verbose: bool, if True print additional info
    min_gene_occurrence: int, remove genes (domain combinations) that occur less than this cutoff
    """
    logger.info(
        f"Removing infrequent genes from the clusters, keeping only"
        f" those that occur in at least {min_gene_occurrence} clusters."
    )
    bgcs = read_clusters_and_remove_empty(in_file_path, min_genes, verbose)

    prev_num_clusters = -1
    prev_num_genes = -1

    filtered_bgcs = bgcs.copy()
    infreq_genes = set()
    iteration = 0
    total_bgcs = len(bgcs)
    while True:
        iteration += 1
        gene_counter = Counter(
            g for genes in filtered_bgcs.values() for g in genes if g != ("-",)
        )
        newly_infreq = set(
            g for g, cnt in gene_counter.items() if cnt < min_gene_occurrence
        )
        infreq_genes.update(newly_infreq)

        logger.info(
            f"[Iteration {iteration}] Found {len(newly_infreq)} new "
            f"infrequent genes (total: {len(infreq_genes)})"
        )

        # Remove infrequent genes and remove clusters with too few genes
        new_filtered_bgcs = OrderedDict()
        n_bgcs_removed = 0
        for bgc_id, genes in filtered_bgcs.items():
            filtered_genes = [("-",) if g in infreq_genes else g for g in genes]
            n_filtered_genes = count_non_empty_genes(filtered_genes)
            if n_filtered_genes < min_genes:
                n_bgcs_removed += 1
                logger.debug(
                    f"[Iteration {iteration}] Excluding {bgc_id}: only "
                    f"{n_filtered_genes} genes with domain hits after "
                    f"removing infrequent genes (min {min_genes})."
                )
            else:
                new_filtered_bgcs[bgc_id] = filtered_genes

        # Check for convergence
        if len(new_filtered_bgcs) == prev_num_clusters:
            logger.info("No further BGCs removed in last iteration. Converged.")
            break
        
        prev_num_clusters = len(new_filtered_bgcs)
        prev_num_genes = len(gene_counter)
        filtered_bgcs = new_filtered_bgcs

    logger.info(
        f"Removed {total_bgcs - len(filtered_bgcs)} of {total_bgcs} BGCs, "
        f" due to containing less than {min_genes} non-empty genes."
    )

    write_clusters(filtered_bgcs, out_file_path)
    write_gene_counts(count_gene_occurrences(filtered_bgcs), counts_file_path)
    logger.info(f"Gene filtered clusters written to {out_file_path}")

    return out_file_path

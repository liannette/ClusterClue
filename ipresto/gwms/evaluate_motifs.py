import logging
from pathlib import Path
from ipresto.clusters.tokenize.orchestrator import TokenizeOrchestrator
from ipresto.gwms.detect_motifs import detect_motifs

logger = logging.getLogger(__name__)

# First we need to tokenise the reference clusters (with annotated subclusters)
# Then we detect the motifs in the tokenised clusters for each set of hyperparameters

# then we read the tsv file with the annotated subclusters and calculate precision/recall/F1-score for each subcluster, motifhit pair

def read_annotated_subclusters(filepath: str) -> dict:
    """Reads annotated subclusters from a TSV file.
    
    Returns a dictionary mapping BGC IDs to lists of annotated subclusters (sets of tokenised genes).
    """
    annotated_subclusters = {}
    with open(filepath, "r") as infile:
        for line in infile:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split("\t")
            bgc_id = parts[0]
            genes = parts[1].split(",")  # assuming genes are comma-separated
            tokenized_genes = set([';'.join(gene.split()) for gene in genes])  # tokenise genes
            if bgc_id not in annotated_subclusters:
                annotated_subclusters[bgc_id] = []
            annotated_subclusters[bgc_id].append(tokenized_genes)
    return annotated_subclusters


def select_best_motif_set(
    motifs_filepaths: list,
    output_dirpath: str,
    reference_subclusters_filepath: str,
    reference_gbks_dirpath: str,
    hmm_file_path: str,
    max_domain_overlap: float,
    cores: int,
    verbose: bool,
    log_queue,
    ):
    """Selects the best motif set based on F1-score against annotated subclusters."""

    output_dirpath.mkdir(parents=True, exist_ok=True)

    logger.info("Reading annotated subclusters")
    annotated_subclusters = read_annotated_subclusters(reference_subclusters_filepath)

    # Tokenize the genes of the clusters into protein domain combinations
    logger.info(f"Preprocessing BGCs containing annotated subclusters from {reference_gbks_dirpath}")
    clusters_file_path = TokenizeOrchestrator().run(
        clusters_file_path=output_dirpath / "clusters.tsv",
        gene_counts_file_path=output_dirpath / "gene_counts.tsv",
        gbks_dir_path=reference_gbks_dirpath,
        hmm_file_path=hmm_file_path,
        exclude_name=[],
        include_contig_edge_clusters=True,
        min_genes=0,
        max_domain_overlap=max_domain_overlap,
        cores=cores,
        verbose=verbose,
        log_queue=log_queue,
    )

    for motifs_filepath in motifs_filepaths:
        logger.info(f"Detecting motifs using motif file {motifs_filepath}")
        motif_hits_filepath = output_dirpath / f"motif_hits_{Path(motifs_filepath).stem}.tsv"
        motif_hits = detect_motifs(
            clusters_filepath=clusters_file_path,
            motifs_filepath=motifs_filepath,
            output_filepath=motif_hits_filepath,
        )



    # logger.info(f"Best motif set is {best_motif_filepath} with F1-score: {best_f1:.4f}")

    # # Copy the best motif file to the output filepath
    # import shutil
    # shutil.copy(best_motif_filepath, output_filepath)
    # logger.info(f"Wrote best motif set to {output_filepath}")


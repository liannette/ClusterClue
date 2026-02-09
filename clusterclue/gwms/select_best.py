import logging
import pandas as pd
from pathlib import Path
from clusterclue.gwms.detect_motifs import detect_motifs, write_motif_hits, parse_clusters_file, parse_motifs_file
from clusterclue.evaluate.evaluate_hits import (
    get_best_hits, 
    calculate_evaluation,
    write_motif_evaluation,
    read_reference_subclusters_and_tokenize_genes
)

logger = logging.getLogger(__name__)



def select_best_motif_set(
    output_dirpath: str | Path,
    gwms_dirpath: str | Path,
    reference_subclusters_filepath: str | Path,
    clusters_filepath: str | Path,
    ):
    """Selects the best motif set based on F1-score against annotated subclusters."""

    logger.info("Reading annotated subclusters file and adding tokenized genes")
    
    clusters = parse_clusters_file(clusters_filepath)
    domain_hits_filepath = clusters_filepath.parent / "all_domain_hits.txt"
    ref_subclusters = read_reference_subclusters_and_tokenize_genes(
        reference_subclusters_filepath, domain_hits_filepath
    )

    # Detect motifs and evaluate motif hits for each motif file
    evaluation_scores = list()
    for motifs_filepath in sorted(Path(gwms_dirpath).iterdir()):
        motif_set_id = motifs_filepath.stem
        motif_gwms = parse_motifs_file(motifs_filepath)
        motif_hits = detect_motifs(clusters, motif_gwms)

        # Evaluate motif hits against annotated subclusters
        best_hits = get_best_hits(ref_subclusters, motif_hits)
        avg_overlap_score, avg_penalized_overlap_score = calculate_evaluation(best_hits)
        logger.info(f"Motif_set: {motif_set_id}, Mean Overlap Score: {avg_overlap_score}, Mean Redundancy Penalized Overlap Score: {avg_penalized_overlap_score}")
        
        evaluation_scores.append({
            "best_hits": best_hits,
            "mean_overlap_score": avg_overlap_score,
            "mean_redundancy_penalised_overlap_score": avg_penalized_overlap_score,
            "motif_set_id": motif_set_id,
            "motif_file": motifs_filepath,
            "motif_hits": motif_hits,
        })

    # Save final scores to a tsv file
    out_file_path = output_dirpath / "evaluation_scores.tsv"
    pd.DataFrame(evaluation_scores)[["motif_set_id", "mean_overlap_score", "mean_redundancy_penalised_overlap_score"]].to_csv(out_file_path, sep="\t", index=False)
    logger.info(f"Wrote all motif set evaluation scores to {out_file_path}")

    # Select the best motif set based on penalized F1-score
    best_motif_set = max(evaluation_scores, key=lambda x: x["mean_redundancy_penalised_overlap_score"])
    logger.info(f"Best motif set: {best_motif_set['motif_set_id']}, Mean Overlap Score (MOS): {best_motif_set['mean_overlap_score']:.4f}, Mean Redundancy Penalised Overlap Score (MRPOS): {best_motif_set['mean_redundancy_penalised_overlap_score']:.4f}")

    #write_evaluation_results(ref_subclusters, best_motif_set, output_dirpath)

    # output files
    motif_gwms_filepath = output_dirpath / "motif_gwms.txt"
    motif_hits_filepath = output_dirpath / "motif_hits.tsv"
    evaluation_best_hits_filepath = output_dirpath / "ref_subclusters_best_hits.tsv"

    # Write best motifs to final output file
    motif_gwms_filepath.write_text(best_motif_set["motif_file"].read_text())
    logger.info(f"Wrote best motif set to {motif_gwms_filepath}")

    # Save motif hits to a tsv file
    motifs = parse_motifs_file(motif_gwms_filepath)
    write_motif_hits(best_motif_set["motif_hits"], motifs, motif_hits_filepath)
    logger.info(f"Wrote motif hits of the best motif set to {motif_hits_filepath}")

    # Save best motif set hits to a tsv file
    write_motif_evaluation(ref_subclusters, best_motif_set["best_hits"], evaluation_best_hits_filepath)
    logger.info(f"Wrote the evaluation details of the best motif set to {evaluation_best_hits_filepath}")

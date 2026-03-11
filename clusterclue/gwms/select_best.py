import logging
import re
import pandas as pd
from pathlib import Path
from clusterclue.gwms.detect_motifs import detect_motifs, write_motif_hits, parse_clusters_file, parse_motifs_file
from clusterclue.evaluate.evaluate_hits import (
    assign_best_hit,
    write_motif_evaluation,
    read_reference_subclusters_and_tokenize_genes
)

logger = logging.getLogger(__name__)


def select_best_motif_set(
    output_dirpath: str | Path,
    gwms_dirpath: str | Path,
    reference_subclusters_filepath: str | Path,
    clusters_filepath: str | Path,
    overlap_penalty_alpha: float = 0.5,
    overlap_penalty_beta: float = 2.0
    ):
    """Selects the best motif set based on overlap score against annotated subclusters.
    
    overlap_penalty_alpha: Penalty factor for redundant hits per subcluster (0=no penalty).
        Higher values penalize more. Uses harmonic decay: penalty = 1/(1 + α(n-1)).
        default is 0.5, which moderately penalizes multiple hits per subcluster.
    """
    # Convert to Path objects if they are strings
    output_dirpath = Path(output_dirpath)
    gwms_dirpath = Path(gwms_dirpath)
    reference_subclusters_filepath = Path(reference_subclusters_filepath)
    clusters_filepath = Path(clusters_filepath)


    logger.info("Reading annotated subclusters file and adding tokenized genes")

    clusters = parse_clusters_file(clusters_filepath)
    domain_hits_filepath = clusters_filepath.parent / "all_domain_hits.txt"
    ref_subclusters = read_reference_subclusters_and_tokenize_genes(
        reference_subclusters_filepath, domain_hits_filepath
    )

    # Detect motifs and evaluate motif hits for each motif file
    evaluation_scores = list()
    for motifs_filepath in sorted(gwms_dirpath.iterdir()):
        motif_set_id = motifs_filepath.stem
        motif_gwms = parse_motifs_file(motifs_filepath)
        motif_hits = detect_motifs(clusters, motif_gwms)

        # Evaluate motif hits against annotated subclusters
        best_hits = pd.DataFrame(
            ref_subclusters.apply(
                lambda row: assign_best_hit(
                    row, motif_hits, 
                    alpha=overlap_penalty_alpha, beta=overlap_penalty_beta
                ), axis=1
            ).tolist()
        )
        mean_overlap_score = best_hits["overlap_score"].mean()
        mean_penalized_score = best_hits["penalized_score"].mean()

        logger.info(
            f"Motif_set: {motif_set_id}, "
            f"n_motifs: {len(motif_gwms)}, "
            f"Score: {round(mean_overlap_score, 4)}, "
            f"Penalized Score (alpha={overlap_penalty_alpha}, beta={overlap_penalty_beta}): "
            f"{round(mean_penalized_score, 4)}"
        )
        
        # extract hyperparameters from motif_set_id using regex
        keys = ['k', 'mm', 'mgc', 'ct', 'mgp']
        pattern = r'_k(\d+)_mm(\d+)_mgc(\d+)_ct(\d+)_mgp(\d+)'
        hyperparameters = {k: int(v) for k, v in zip(keys, re.search(pattern, motif_set_id).groups())}

        evaluation_scores.append({
            "best_hits": best_hits,
            "mean_overlap_score": mean_overlap_score,
            "mean_penalized_score": mean_penalized_score,
            "motif_set_id": motif_set_id,
            "hyperparameters": hyperparameters,
            "n_motifs": len(motif_gwms),
            "motif_file": motifs_filepath,
            "motif_hits": motif_hits,
        })

    # Save final scores to a tsv file
    out_file_path = output_dirpath / "evaluation_scores.tsv"
    evaluation_scores_df = pd.DataFrame(evaluation_scores)[["motif_set_id", "n_motifs", "mean_overlap_score", "mean_penalized_score"]]
    # round
    evaluation_scores_df["mean_overlap_score"] = evaluation_scores_df["mean_overlap_score"].round(4)
    evaluation_scores_df["mean_penalized_score"] = evaluation_scores_df["mean_penalized_score"].round(4)
    evaluation_scores_df.to_csv(out_file_path, sep="\t", index=False)
    logger.info(f"Wrote all motif set evaluation scores to {out_file_path}")

    # Select the best motif set based on penalized F1-score
    best_motif_set = max(evaluation_scores, key=lambda x: x["mean_penalized_score"], )
    best_motif_set = max(
        evaluation_scores,
        key=lambda x: (
            x["mean_penalized_score"],
            x["hyperparameters"]["mgc"],
            x["hyperparameters"]["ct"],
            x["hyperparameters"]["mgp"],
            x["hyperparameters"]["mm"],
            x["n_motifs"],
        ),
    )
    logger.info(
        f"Best motif set with best penalized score: {best_motif_set['motif_set_id']} ({best_motif_set['n_motifs']} motifs) "
        f"Mean Overlap Score: {round(best_motif_set['mean_overlap_score'], 4)}, "
        f"Mean Penalized Score (alpha={overlap_penalty_alpha}, beta={overlap_penalty_beta}): "
        f"{round(best_motif_set['mean_penalized_score'], 4)}"
        )

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

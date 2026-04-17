from pathlib import Path
import logging
import pandas as pd
import json
from dataclasses import dataclass
from typing import Dict, List
from clusterclue.evaluate.evaluate_hits import read_reference_subclusters_and_tokenize_genes
from clusterclue.evaluate.evaluate_hits import assign_best_hit, write_motif_evaluation
from clusterclue.presto_stat.orchestrator import StatOrchestrator
from clusterclue.presto_top.orchestrator import TopOrchestrator

logger = logging.getLogger(__name__)


@dataclass
class PrestoHit:
    bgc_id: str
    motif_id: str
    score: float
    tokenized_genes: set


def load_presto_top_subclusters(filepath: str) -> Dict[str, List[PrestoHit]]:
    """
    Load and parse PRESTO-TOP results from a given file.

    The file is expected to have lines starting with '>' indicating a new BGC (Biosynthetic Gene Cluster) ID,
    followed by lines containing detected subcluster motifs. Lines starting with 'class=' are ignored.

    Args:
        filepath (str): The path to the file containing subcluster data.

    Returns:
        Dict[str, List[PrestoHit]]: A dictionary where each key is a BGC ID and the value is a list of PrestoHit objects.
            Each PrestoHit object contains:
                - 'bgc_id' (str): The BGC ID
                - 'motif_id' (str): The motif ID prefixed with 'T'
                - 'score' (float): The score associated with the hit (set to None)
                - 'tokenized_genes' (set): A set of gene names associated with the subcluster.
    """
    detected_subclusters = dict()
    with open(filepath, "r") as infile:
        for line in infile:
            line = line.rstrip()
            if line.startswith(">"):
                bgc_id = line[1:]
                detected_subclusters[bgc_id] = []
            elif line.startswith("class="):
                pass  # skip the line
            else:
                cols = line.split("\t")
                subcluster_id = f"T{cols[0]}"
                genes = [m.split(":")[0] for m in cols[3].split(",")]
                detected_subclusters[bgc_id].append(
                    PrestoHit(bgc_id=bgc_id, motif_id=subcluster_id, score=None, tokenized_genes=set(genes))
                )
    return detected_subclusters


def load_presto_stat_subclusters(
    filepath: str,
) -> Dict[str, List[PrestoHit]]:
    """
    Load and parse PRESTO-STAT results from a given file.

    This function reads a file containing detected subclusters and parses it into a dictionary.
    Each key in the dictionary is a BGC  ID, and the value is a list of subcluster dictionaries.
    Each subcluster dictionary contains an 'id' and a list of 'genes'.

    Args:
        filepath (str): The path to the Presto statistics file.

    Returns:
        Dict[str, List[PrestoHit]]: A dictionary where each key is a BGC ID and the value is a list of PrestoHit objects.
            Each PrestoHit object contains:
                - 'bgc_id' (str): The BGC ID
                - 'motif_id' (str): The module ID 
                - 'score' (float): The score associated with the hit (if available)
                - 'tokenized_genes' (set): A set of gene names associated with the subcluster.
    """
    detected_subclusters = dict()
    with open(filepath, "r") as infile:
        for line in infile:
            # header lines
            if line.startswith(">"):
                bgc_id = line.rstrip()[1:]
                detected_subclusters[bgc_id] = list()
            else:
                cols = line.rstrip().split("\t")
                module_id = cols[0]
                genes = cols[1].split(",")
                detected_subclusters[bgc_id].append(
                    PrestoHit(bgc_id=bgc_id, motif_id=module_id, score=None, tokenized_genes=set(genes))
                )
    return detected_subclusters


def get_best_hits(
    ref_subclusters: pd.DataFrame, 
    hits: Dict[str, List[PrestoHit]], 
    alpha: float = 0.25, 
    beta: float = 2.0
) -> pd.DataFrame:
    """
    Apply assign_best_hit to each row of reference subclusters and return as DataFrame.
    
    Args:
        ref_subclusters: DataFrame with annotated subclusters
        hits: Dictionary mapping BGC IDs to lists of PrestoHit objects
        alpha: Penalty strength parameter (default 0.25)
        beta: Penalty growth rate parameter (default 2.0)
    
    Returns:
        DataFrame with best hits for each reference subcluster
    """
    best_hits = ref_subclusters.apply(
        lambda row: assign_best_hit(row, hits, alpha=alpha, beta=beta), 
        axis=1
    )
    return pd.DataFrame(best_hits.tolist())


def calculate_evaluation(ref_subclusters_with_hits: pd.DataFrame) -> tuple:
    """Calculate mean overlap score and mean penalized overlap score."""
    scores = ref_subclusters_with_hits[["overlap_score", "penalized_score"]]
    mean_scores = scores.mean().to_dict()

    m_os = round(mean_scores["overlap_score"], 3)
    m_ps = round(mean_scores["penalized_score"], 3)
    return m_os, m_ps


def evaluate_presto_stat(
    reference_subclusters_filepath: Path,
    reference_clusters_filepath: Path,
    stat_modules: Path,
    output_dirpath: Path,
    overlap_penalty_alpha: float = 0.25,
    overlap_penalty_beta: float = 2.0
) -> dict:
    """Evaluate PRESTO-STAT detected subclusters against a reference set of annotated subclusters.
    
    Returns:
        dict: Results dictionary with evaluation metrics and file paths
    """

    # run presto stat on reference clusters
    StatOrchestrator().run(
        out_dir_path=output_dirpath,
        cluster_file=reference_clusters_filepath,
        stat_modules_file_path=stat_modules,
        pval_cutoff=None,
        n_families_range=[],
        min_genes_per_bgc=0,
        cores=1,
        verbose=False
        )
    
    # load reference subclusters
    domain_hits_filepath = Path(reference_clusters_filepath).parent / "all_domain_hits.txt"
    ref_subclusters = read_reference_subclusters_and_tokenize_genes(
        reference_subclusters_filepath, domain_hits_filepath
    )

    # load detected subclusters
    detected_subclusters_filepath = Path(output_dirpath) / "detected_stat_modules.txt"
    hits = load_presto_stat_subclusters(detected_subclusters_filepath)

    # evaluate
    eval_df = get_best_hits(ref_subclusters, hits, alpha=overlap_penalty_alpha, beta=overlap_penalty_beta)
    
    # Save to csv
    eval_best_hits_filepath = Path(output_dirpath) / "best_hits_PRESTO-STAT.tsv"
    eval_df.to_csv(eval_best_hits_filepath, sep="\t", index=False)

    # merge for final evaluation
    ref_subclusters_with_hits = ref_subclusters.merge(eval_df, on="subcluster_id")
    avg_overlap_score, avg_penalized_overlap_score = calculate_evaluation(ref_subclusters_with_hits)
    
    logger.info(f"PRESTO-STAT Mean Overlap Score: {avg_overlap_score}, Mean Penalized Overlap Score: {avg_penalized_overlap_score}")

    # Calculate statistics
    n_detected_subclusters = sum(len(hit_list) for hit_list in hits.values())
    n_bgcs_with_hits = len(hits)
    
    # Store results
    results = {'PRESTO-STAT': {
        'n_detected_subclusters': int(n_detected_subclusters),
        'n_bgcs_with_hits': int(n_bgcs_with_hits),
        'mean_overlap_score': float(avg_overlap_score),
        'mean_penalized_score': float(avg_penalized_overlap_score),
        'alpha': float(overlap_penalty_alpha),
        'beta': float(overlap_penalty_beta),
        'eval_hits_filepath': str(detected_subclusters_filepath),
        'eval_best_hits_filepath': str(eval_best_hits_filepath),
    }}

    return results


def evaluate_presto_top(
    reference_subclusters_filepath: Path,
    reference_clusters_filepath: Path,
    top_model_file_path: Path,
    number_of_topics: int,
    output_dirpath: Path,
    overlap_penalty_alpha: float = 0.25,
    overlap_penalty_beta: float = 2.0
) -> dict:
    """Evaluate PRESTO-TOP detected subclusters against a reference set of annotated subclusters.
    
    Returns:
        dict: Results dictionary with evaluation metrics and file paths
    """

    # run presto top on reference clusters
    TopOrchestrator().run(
        out_dir_path=output_dirpath,
        cluster_file=reference_clusters_filepath,
        top_model_file_path=top_model_file_path,
        topics=number_of_topics,
        amplify=False,
    )

    # load reference subclusters
    domain_hits_filepath = Path(reference_clusters_filepath).parent / "all_domain_hits.txt"
    ref_subclusters = read_reference_subclusters_and_tokenize_genes(
        reference_subclusters_filepath, domain_hits_filepath
    )
    
    # load detected subclusters
    detected_subclusters_filepath = Path(output_dirpath) / "bgc_topics_filtered.txt"
    hits = load_presto_top_subclusters(detected_subclusters_filepath)

    # evaluate
    eval_df = get_best_hits(ref_subclusters, hits, alpha=overlap_penalty_alpha, beta=overlap_penalty_beta)
    
    # Save to csv
    eval_best_hits_filepath = Path(output_dirpath) / "best_hits_PRESTO-TOP.tsv"
    eval_df.to_csv(eval_best_hits_filepath, sep="\t", index=False)

    # merge for final evaluation
    ref_subclusters_with_hits = ref_subclusters.merge(eval_df, on="subcluster_id")
    avg_overlap_score, avg_penalized_overlap_score = calculate_evaluation(ref_subclusters_with_hits)
    
    logger.info(f"PRESTO-TOP Mean Overlap Score: {avg_overlap_score}, Mean Penalized Overlap Score: {avg_penalized_overlap_score}")

    # Calculate statistics
    n_detected_subclusters = sum(len(hit_list) for hit_list in hits.values())
    n_bgcs_with_hits = len(hits)
    
    # Store results
    results = {'PRESTO-TOP': {
        'n_detected_subclusters': int(n_detected_subclusters),
        'n_bgcs_with_hits': int(n_bgcs_with_hits),
        'mean_overlap_score': float(avg_overlap_score),
        'mean_penalized_score': float(avg_penalized_overlap_score),
        'alpha': float(overlap_penalty_alpha),
        'beta': float(overlap_penalty_beta),
        'eval_hits_filepath': str(detected_subclusters_filepath),
        'eval_best_hits_filepath': str(eval_best_hits_filepath),
    }}

    return results
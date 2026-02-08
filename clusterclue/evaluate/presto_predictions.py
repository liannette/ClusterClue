from pathlib import Path
import logging
from dataclasses import dataclass
from typing import Dict, List
from clusterclue.evaluate.evaluate_hits import read_reference_subclusters_and_tokenize_genes
from clusterclue.evaluate.evaluate_hits import calculate_evaluation, get_best_hits, write_motif_evaluation
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


def evaluate_presto_stat(
    reference_subclusters_filepath: Path,
    reference_clusters_filepath: Path,
    stat_modules: Path,
    output_dirpath: Path):
    """Evaluate PRESTO-STAT detected subclusters against a reference set of annotated subclusters."""

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
    
    # load detected subclusters
    detected_subclusters_filepath = Path(output_dirpath) / "detected_stat_modules.txt"
    hits = load_presto_stat_subclusters(detected_subclusters_filepath)

    # load reference subclusters
    domain_hits_filepath = Path(reference_clusters_filepath).parent / "all_domain_hits.txt"
    ref_subclusters = read_reference_subclusters_and_tokenize_genes(
        reference_subclusters_filepath, domain_hits_filepath
    )

    # evaluate
    best_hits = get_best_hits(ref_subclusters, hits)
    avg_overlap_score, avg_penalized_overlap_score = calculate_evaluation(best_hits)
    
    logger.info(f"PRESTO-STAT Mean Overlap Score: {avg_overlap_score}, Mean Redundancy Penalized Overlap Score: {avg_penalized_overlap_score}")

    output_filepath = Path(output_dirpath) / "ref_subclusters_best_hits.tsv"
    write_motif_evaluation(
        ref_subclusters,
        best_hits,
        output_filepath,
    )
    logger.info(f"PRESTO-STAT best hits to reference subclusters results written to: {output_filepath}")


def evaluate_presto_top(
    reference_subclusters_filepath: Path,
    reference_clusters_filepath: Path,
    top_model_file_path: Path,
    number_of_topics: int,
    output_dirpath: Path
    ):
    """Evaluate PRESTO-TOP detected subclusters against a reference set of annotated subclusters."""

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
    best_hits = get_best_hits(ref_subclusters, hits)
    avg_overlap_score, avg_penalized_overlap_score = calculate_evaluation(best_hits)
    
    logger.info(f"PRESTO-TOP Mean Overlap Score: {avg_overlap_score}, Mean Redundancy Penalized Overlap Score: {avg_penalized_overlap_score}")

    output_filepath = Path(output_dirpath) / "ref_subclusters_best_hits.tsv"
    write_motif_evaluation(
        ref_subclusters,
        best_hits,
        output_filepath,
    )   
    logger.info(f"PRESTO-TOP best hits to reference subclusters results written to: {output_filepath}")
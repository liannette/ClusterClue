import logging
import pandas as pd

from typing import Dict, List
from pathlib import Path 
from clusterclue.classes.hits import MotifHit, PrestoHit


logger = logging.getLogger(__name__)


def load_domain_hits(dom_hits_file: str) -> Dict[str, Dict[str, List[str]]]:
    domain_hits = dict()
    with open(dom_hits_file, "r") as infile:
        infile.readline()  # skip the header
        for line in infile:
            row = line.split("\t")
            bgc = row[0]
            protein_id = row[2]
            domain = row[6]
            if bgc not in domain_hits:
                domain_hits[bgc] = {protein_id: [domain]}
            elif protein_id not in domain_hits[bgc]:
                domain_hits[bgc][protein_id] = [domain]
            else:
                domain_hits[bgc][protein_id].append(domain)
    return domain_hits


def get_tokenized_gene(
    mibig_acc: str,
    protein_ids: List[str],
    domain_hits: Dict[str, Dict[str, List[str]]],
) -> List[str]:
    """Get domain combinations for a list of protein IDs."""
    tokenized_genes = []
    for protein_id in protein_ids:
        if (
            mibig_acc in domain_hits
            and protein_id in domain_hits[mibig_acc]
        ):
            tokenized_genes.append(";".join(domain_hits[mibig_acc][protein_id]))
        else:
            tokenized_genes.append("-")
    return tokenized_genes


def read_reference_subclusters_and_tokenize_genes(
    reference_subclusters_filepath: str,
    domain_hits_filepath: str,
) -> pd.DataFrame:
    """Read annotated subclusters and add tokenized genes based on biosynthetic domains."""
    domain_hits = load_domain_hits(domain_hits_filepath)
    subclusters = pd.read_csv(reference_subclusters_filepath, sep="\t")
    subclusters["tokenized_genes"] = subclusters.apply(
        lambda x: get_tokenized_gene(
            x["mibig_acc"], x["protein_ids"].split(";"), domain_hits
        ),
        axis=1,
    )
    return subclusters


def calculate_precision(hit_genes: set, annotated_genes: set) -> float:
    """Calculate ratio between  genes to motif hit genes."""
    return (
        len(hit_genes.intersection(annotated_genes)) / len(hit_genes) 
        if hit_genes
        else 0.0
        )


def calculate_recall(hit_genes: set, annotated_genes: set) -> float:
    """Calculate ratio between overlapping genes and annotated subcluster genes."""
    return (
        len(hit_genes.intersection(annotated_genes)) / len(annotated_genes)
        if annotated_genes
        else 0.0
    )


def calculate_fbeta_score(precision: float, recall: float, beta: float) -> float:
    """Calculate the F-beta score, which weights recall more than precision if beta > 1."""
    if precision == 0 and recall == 0:
        return 0.0
    return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)


def calculate_jaccard(annotated_genes: set, hit_genes: set) -> float:
    """Calculate the Jaccard index as the ratio of overlapping genes to the union of both sets."""
    overlapping_genes = annotated_genes.intersection(hit_genes)
    union_genes = annotated_genes.union(hit_genes)
    return len(overlapping_genes) / len(union_genes)


def calculate_penalized_f1(
    f1: float, n_overlapping_hits: int, alpha: float
) -> float:
    """Calculate the penalized F1 score based on the number of hits with Jaccard index above a threshold.

    The term (n_overlapping_hits−1) ensures no penalty when only one predicted subcluster overlaps 
    (ideal case), with penalties scaling for n_overlapping_hits>1.
    

    Args:
        f1 (float): The F1 score to be penalized.
        n_overlapping_hits (int): Number of of hits with Jaccard index above the threshold.
        alpha (float): Controls the penalty strength, higher values (e.g. 0.5) 
            penalize more than lower values (e.g. 0.1).

    Returns:
        float: The penalized F1 score.
    """
    penalty = 1 / (1 + alpha * (n_overlapping_hits - 1))
    return f1 * penalty


def assign_best_hit(row: pd.Series, hits: Dict[str, List[MotifHit | PrestoHit]]) -> dict:
    """Find the best hit for a given row of annotated subclusters."""
    mibig_acc = row["mibig_acc"]
    annotated_genes = set(row["tokenized_genes"])

    overlapping_hits = []
    tau = 0.1 # jaccard threshold for for a hit to be considered
    for hit in hits.get(mibig_acc, []):
        hit_genes = set(hit.tokenized_genes)
        jaccard = calculate_jaccard(annotated_genes, hit_genes)
        # Skip hits with jaccard below the threshold
        if jaccard < tau:
            continue

        overlap = annotated_genes.intersection(hit_genes)
        recall = calculate_recall(hit_genes, annotated_genes)
        precision = calculate_precision(hit_genes, annotated_genes)
        f1 = calculate_fbeta_score(precision, recall, beta=1)
        overlapping_hits.append(
            {
                "subcluster_id": row["subcluster_id"],
                "motif_id": hit.motif_id,
                "motif_hit_genes": hit.tokenized_genes,
                "n_overlapping_genes": len(overlap),
                "overlapping_genes": list(overlap),
                "jaccard": jaccard,
                "recall": recall,
                "precision": precision,
                "overlap_score": f1,
            }
        )

    if overlapping_hits:
        best_hit = max(overlapping_hits, key=lambda x: x["overlap_score"])
        alpha = 0.25 # moderately penalize multiple motif hits per subcluster
        n_overlapping_hits = len(overlapping_hits)
        penalized_f1 = calculate_penalized_f1(best_hit["overlap_score"], n_overlapping_hits, alpha)
        best_hit["n_overlapping_hits"] = n_overlapping_hits
        best_hit["redundancy_penalised_overlap_score"] = penalized_f1
    else:
        best_hit = {
            "subcluster_id": row["subcluster_id"],
            "motif_id": None,
            "motif_hit_genes": [],
            "n_overlapping_genes": 0,
            "overlapping_genes": [],
            "jaccard": 0.0,
            "recall": 0.0,
            "precision": 0.0,
            "overlap_score": 0.0,
            "n_overlapping_hits": 0,
            "redundancy_penalised_overlap_score": 0.0,
        }

    return best_hit


def get_best_hits(
    ref_subclusters: pd.DataFrame,
    hits: Dict[str, List[MotifHit | PrestoHit]]
) -> pd.DataFrame:
    best_hits = pd.DataFrame(
        ref_subclusters.apply(lambda row: assign_best_hit(row, hits), axis=1).tolist()
    )
    return best_hits


def calculate_evaluation(ref_subclusters_with_hits: pd.DataFrame) -> tuple:
    scores = ref_subclusters_with_hits[["overlap_score", "redundancy_penalised_overlap_score"]]
    mean_scores = scores.mean().to_dict()

    m_os = round(mean_scores["overlap_score"], 3)
    m_rpos = round(mean_scores["redundancy_penalised_overlap_score"], 3)
    return m_os, m_rpos


def write_motif_evaluation(ref_subclusters: pd.DataFrame, best_hits: pd.DataFrame, output_filepath: Path) -> None:
    """Writes the best motif set hits to a tsv file."""
    eval_df = ref_subclusters.merge(best_hits, on="subcluster_id")
    eval_df["tokenized_genes"] = eval_df["tokenized_genes"].apply(lambda x: ";".join(x))
    eval_df["motif_hit_genes"] = eval_df["motif_hit_genes"].apply(lambda x: ";".join(sorted(x)))
    eval_df["overlapping_genes"] = eval_df["overlapping_genes"].apply(lambda x: ";".join(sorted(x)))
    eval_df["jaccard"] = eval_df["jaccard"].round(3)
    eval_df["recall"] = eval_df["recall"].round(3)
    eval_df["precision"] = eval_df["precision"].round(3)
    eval_df["overlap_score"] = eval_df["overlap_score"].round(3)
    eval_df["redundancy_penalised_overlap_score"] = eval_df["redundancy_penalised_overlap_score"].round(3)
    eval_df.to_csv(output_filepath, sep="\t", index=False)

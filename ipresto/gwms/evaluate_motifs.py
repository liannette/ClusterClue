import logging
import pandas as pd
from typing import Dict, List
from pathlib import Path
from ipresto.clusters.tokenize.orchestrator import TokenizeOrchestrator
from ipresto.gwms.detect_motifs import detect_motifs
from ipresto.gwms.detect_motifs import MotifHit

logger = logging.getLogger(__name__)

# First we need to tokenise the reference clusters (with annotated subclusters)
# Then we detect the motifs in the tokenised clusters for each set of hyperparameters

# then we read the tsv file with the annotated subclusters and calculate precision/recall/F1-score for each subcluster, motifhit pair


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


def calculate_precision(overlapping_genes: set, hit_genes: set) -> float:
    """Calculate precision as the ratio of true positive genes to detected subcluster genes."""
    return len(overlapping_genes) / len(hit_genes) if hit_genes else 0.0


def calculate_recall(overlapping_genes: set, annotated_subcluster_genes: set) -> float:
    """Calculate recall as the ratio of true positive genes to annotated subcluster genes."""
    return (
        len(overlapping_genes) / len(annotated_subcluster_genes)
        if annotated_subcluster_genes
        else 0.0
    )


def calculate_fbeta_score(precision: float, recall: float, beta: float) -> float:
    """Calculate the F-beta score, which weights recall more than precision if beta > 1."""
    if precision == 0 and recall == 0:
        return 0.0
    return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)


def calculate_jaccard(annotated_subcluster_genes: set, hit_genes: set) -> float:
    """Calculate the Jaccard index as the ratio of overlapping genes to the union of both sets."""
    overlapping_genes = annotated_subcluster_genes.intersection(hit_genes)
    union_genes = annotated_subcluster_genes.union(hit_genes)
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


def assign_best_hit(row: pd.Series, motif_hits: Dict[str, List[MotifHit]]) -> dict:
    """Find the best hit for a given row of annotated subclusters."""
    mibig_acc = row["mibig_acc"]
    sc_tokenized_gene_set = set(row["tokenized_genes"])

    overlapping_hits = []
    tau = 0.1 # jaccard threshold for for a hit to be considered
    for hit in motif_hits.get(mibig_acc, []):
        overlap = sc_tokenized_gene_set.intersection(hit.tokenized_genes)
        jaccard = calculate_jaccard(sc_tokenized_gene_set, hit.tokenized_genes)
        # Skip hits with jaccard below the threshold
        if jaccard < tau:
            continue

        recall = calculate_recall(overlap, sc_tokenized_gene_set)
        precision = calculate_precision(overlap, hit.tokenized_genes)
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
                "f1": f1,
            }
        )

    if overlapping_hits:
        best_hit = max(overlapping_hits, key=lambda x: x["f1"])
        alpha = 0.25 # moderately penalize multiple motif hits per subcluster
        n_overlapping_hits = len(overlapping_hits)
        penalized_f1 = calculate_penalized_f1(best_hit["f1"], n_overlapping_hits, alpha)
        best_hit["n_overlapping_hits"] = n_overlapping_hits
        best_hit["penalized_f1"] = penalized_f1
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
            "f1": 0.0,
            "n_overlapping_hits": 0,
            "penalized_f1": 0.0,
        }

    return best_hit


def add_motif_hits_to_reference_subclusters(
    ref_subclusters: pd.DataFrame,
    motif_hits: Dict[str, List[MotifHit]]
) -> pd.DataFrame:
    best_hits = pd.DataFrame(
        ref_subclusters.apply(lambda row: assign_best_hit(row, motif_hits), axis=1).tolist()
    )
    return best_hits


def calculate_evaluation(ref_subclusters_with_hits: pd.DataFrame) -> tuple:
    scores = ref_subclusters_with_hits[["f1", "penalized_f1"]]
    avg_scores = scores.mean().to_dict()

    avg_f1 = round(avg_scores["f1"], 3)
    avg_penalized_f1 = round(avg_scores["penalized_f1"], 3)
    return avg_f1, avg_penalized_f1


def write_motif_file(ref_subclusters: pd.DataFrame, best_motif_set: dict, output_filepath: Path) -> None:
    """Writes the best motif set hits to a tsv file."""
    eval_df = ref_subclusters.merge(best_motif_set["best_hits"], on="subcluster_id")
    eval_df["tokenized_genes"] = eval_df["tokenized_genes"].apply(lambda x: ";".join(x))
    eval_df["motif_hit_genes"] = eval_df["motif_hit_genes"].apply(lambda x: ";".join(sorted(x)))
    eval_df["overlapping_genes"] = eval_df["overlapping_genes"].apply(lambda x: ";".join(sorted(x)))
    eval_df["jaccard"] = eval_df["jaccard"].round(3)
    eval_df["recall"] = eval_df["recall"].round(3)
    eval_df["precision"] = eval_df["precision"].round(3)
    eval_df["f1"] = eval_df["f1"].round(3)
    eval_df["penalized_f1"] = eval_df["penalized_f1"].round(3)
    eval_df.to_csv(output_filepath, sep="\t", index=False)


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

    # Tokenize the genes of the clusters into protein domain combinations
    logger.info(f"Preprocessing BGCs containing annotated subclusters from {reference_gbks_dirpath}")
    preprocess_dir_path = output_dirpath / "preprocess_reference_clusters"
    preprocess_dir_path.mkdir(parents=True, exist_ok=True)
    clusters_file_path = TokenizeOrchestrator().run(
        clusters_file_path=preprocess_dir_path / "clusters.tsv",
        gene_counts_file_path=preprocess_dir_path / "gene_counts.tsv",
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

    logger.info("Reading annotated subclusters file and adding tokenized genes")
    domain_hits_filepath = clusters_file_path.parent / "all_domain_hits.txt"
    ref_subclusters = read_reference_subclusters_and_tokenize_genes(
        reference_subclusters_filepath, domain_hits_filepath
    )

    # Detect motifs and evaluate motif hits for each motif file
    evaluation_scores = list()
    motif_hits_dirpath = output_dirpath / "motif_hits"
    motif_hits_dirpath.mkdir(parents=True, exist_ok=True)
    for motifs_filepath in motifs_filepaths:
        motif_hits_filepath = motif_hits_dirpath / f"motif_hits_{Path(motifs_filepath).stem}.tsv"
        motif_hits = detect_motifs(
            clusters_filepath=clusters_file_path,
            motifs_filepath=motifs_filepath,
            output_filepath=motif_hits_filepath,
        )

        # Evaluate motif hits against annotated subclusters
        logger.info("Evaluating motif hits against annotated subclusters")
        best_hits = add_motif_hits_to_reference_subclusters(ref_subclusters, motif_hits)
        avg_f1, avg_penalized_f1 = calculate_evaluation(best_hits)
        logger.info(f"Average F1-score: {avg_f1}, Average Penalized F1-score: {avg_penalized_f1}")
        
        evaluation_scores.append({
            "best_hits": best_hits,
            "f1": avg_f1,
            "penalized_f1": avg_penalized_f1,
            "motif_set": motifs_filepath.stem,
            "motif_file": motifs_filepath,
        })

    # Save final scores to a tsv file
    out_file_path = output_dirpath / "evaluation_scores.tsv"
    pd.DataFrame(evaluation_scores)[["f1", "penalized_f1", "motif_set"]].to_csv(out_file_path, sep="\t", index=False)
    logger.info(f"Wrote all motif set evaluation scores to {out_file_path}")

    # Select the best motif set based on penalized F1-score
    best_motif_set = max(evaluation_scores, key=lambda x: x["penalized_f1"])
    logger.info(f"Best motif set is {best_motif_set['motif_set']} with Penalized F1-score: {best_motif_set['penalized_f1']:.4f}")
    
    # Save best motif set hits to a tsv file
    best_motif_set_hits_filepath = output_dirpath / "best_motif_set_hits.tsv"
    write_motif_file(ref_subclusters, best_motif_set, best_motif_set_hits_filepath)
    logger.info(f"Wrote best motif set hits to {best_motif_set_hits_filepath}")

    return Path(best_motif_set["motif_file"])
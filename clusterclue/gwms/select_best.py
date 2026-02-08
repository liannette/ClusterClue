import logging
import pandas as pd
import csv
from pathlib import Path
from clusterclue.gwms.detect_motifs import detect_motifs, write_motif_hits, parse_clusters_file, parse_motifs_file
from subsketch.reports import generate_html_for_annotated_subcluster
from subsketch.io import read_compounds
from subsketch.loaders import load_mibig_bgc
from clusterclue.evaluate.evaluate_hits import (
    get_best_hits, 
    calculate_evaluation,
    write_motif_evaluation,
    read_reference_subclusters_and_tokenize_genes
)
import subsketch as subsk


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


def visualize_evaluation_results(
    motifs_dirpath, 
    ref_clusters_filepath,
    ref_gbks_dirpath,
    mibig_compounds_filepath, 
    ) -> None:
    motifs_dirpath = Path(motifs_dirpath)
    evaluation_best_hits_filepath = motifs_dirpath /"ref_subclusters_best_hits.tsv"
    motif_gwms_filepath = motifs_dirpath / "motif_gwms.txt"
    motif_hits_filepath = motifs_dirpath / "motif_hits.tsv"
    domain_hits_filepath = Path(ref_clusters_filepath).parent / "all_domain_hits.txt"

    session = subsk.SubSketchSession(
        motifs_file=motif_gwms_filepath,
        genbank_dir=ref_gbks_dirpath,
        domain_hits_file=domain_hits_filepath,
        motif_hits_file=motif_hits_filepath,
        compounds_file=mibig_compounds_filepath,
    )
    session.load()

    mibig_compounds = read_compounds(mibig_compounds_filepath)

    subclusters = list()
    with open(evaluation_best_hits_filepath, "r") as infile:
        reader = csv.DictReader(infile, delimiter="\t")
        for row in reader:
            subclusters.append({
                "id": row["subcluster_id"],
                "bgc_id": row["mibig_acc"],
                "compound_name": row["bgc_product"],
                "substructure_name": row["substructure"],
                "substructure_class": row["substructure class"],
                "substructure_smiles": row["substructure smiles"],
                "genes": row["genes"].split(";"),
                "protein_ids": row["protein_ids"].split(";"),
                "pathway_quality": row["pathway quality"],
                "pubmed_id": [],
                "orig_seq": "N/A",
            })

    out_html_dirpath = motifs_dirpath / "ref_subclusters_hits"
    out_html_dirpath.mkdir(exist_ok=True, parents=True)

    for subcluster in subclusters:
        #html_content += f"<h2>Subcluster {subcluster['id']} in BGC {subcluster['bgc_id']}</h2>\n"
        html_content = ""

        bgc_id = subcluster["bgc_id"]
        bgc_data = load_mibig_bgc(Path(ref_gbks_dirpath) / f"{bgc_id}.gbk")
        bgc_compounds = mibig_compounds.get(bgc_id, [])

        html_content += generate_html_for_annotated_subcluster(
            subcluster=subcluster,
            bgc_data=bgc_data,
            compounds=bgc_compounds,
            gene_arrow_scaling=60,
            )

        html_content += session.html_report_for_bgc(
            bgc_id=bgc_id,
            include_title=False,
            include_bgc_plot=False,
            include_compound_plots=False,
            include_motif_plots=True,
            gene_arrow_scaling=60, 
        )

        out_html = out_html_dirpath / f"{subcluster['id']}_hits.html"
        with open(out_html, "w") as outfile:
            outfile.write(html_content)
        logger.info(f"Wrote motif evaluation visualization to {out_html}")
    
from pathlib import Path
import logging
import csv
import subsketch as subsk
from subsketch.reports import generate_html_for_annotated_subcluster
from subsketch.io import read_compounds
from subsketch.loaders import load_mibig_bgc

logger = logging.getLogger(__name__)

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
    
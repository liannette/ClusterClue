from pathlib import Path
from Bio import SeqIO
from clusterclue.visualize.bgc import draw_bgc
from clusterclue.visualize.molecule import draw_compounds
from clusterclue.visualize.utils import (
    read_txt,
    read_detected_motifs,
    read_dom_hits,
    read_color_domains_file,
    read_compounds,
    write_combined_html,
    write_separate_htmls,
)


def get_subcluster_header(motif_hit):
    text = (
        f"Motif: {motif_hit['motif_id']} "
        f"(n: {motif_hit['n_matches']}, threshold: {motif_hit['threshold']}), "
        f"score: {motif_hit['score']}, n_genes: {len(motif_hit['genes'])}"
    )
    return f"<div><h3>{text}</h3></div>\n"


def run_visualisation(
    outfile,
    gbks_filepath,
    dom_hits_filepath,
    domain_colors_filepath,
    detected_motifs_filepath,
    compounds_filepath,
    separate_html_files,
    verbose,
):
    if verbose:
        print("\nVisualising detected motifs...")

    # Read input files
    bgc_gbk_paths = [Path(path) for path in read_txt(gbks_filepath)]
    dom_hits = read_dom_hits(dom_hits_filepath)
    domain_colors = read_color_domains_file(domain_colors_filepath)
    detected_motifs = (
        read_detected_motifs(detected_motifs_filepath)
        if detected_motifs_filepath
        else None
    )
    compounds = read_compounds(compounds_filepath) if compounds_filepath else None

    bgc_htmls = dict()
    for bgc_path in bgc_gbk_paths:

        # Get BGC info
        bgc_id = bgc_path.stem
        bgc_seq_record = list(SeqIO.parse(bgc_path, "genbank"))[0]
        bgc_cds_features = [f for f in bgc_seq_record.features if f.type == "CDS"]

        html_content = ""

        # header with BGC ID
        html_content += f"<h1>{bgc_id}</h1>\n"

        # Draw the molecule structure if available
        if compounds:
            html_content += draw_compounds(compounds.get(bgc_id, []))

        # Draw the full BGC
        html_content += draw_bgc(
            bgc_id=bgc_id,
            bgc_length=len(bgc_seq_record),
            cds_features=bgc_cds_features,
            domain_hits=dom_hits,
            domain_colors=domain_colors,
        )

        # Draw the detected motifs
        for motif_hit in detected_motifs.get(bgc_id, []):
            html_content += get_subcluster_header(motif_hit)
            html_content += draw_bgc(
                bgc_id=bgc_id,
                bgc_length=len(bgc_seq_record),
                cds_features=bgc_cds_features,
                domain_hits=dom_hits,
                motif_hit=motif_hit,
                domain_colors=domain_colors,
            )

        bgc_htmls[bgc_id] = html_content
    
    # Write output HTML(s)
    out_path = Path(outfile)
    if separate_html_files:
        out_path = out_path.parent / out_path.stem
        out_path.mkdir(parents=True, exist_ok=True)
        write_separate_htmls(out_path, bgc_htmls)
    else:
        write_combined_html(out_path, bgc_htmls)
    if verbose:
        print(f"  Wrote output to {out_path}")

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
)


def run_visualisation(
    outfile,
    gbks_filepath,
    dom_hits_filepath,
    domain_colors_filepath,
    detected_motifs_filepath=None,
    compounds_filepath=None,
    verbose=False,
):
    # Read BGC paths
    bgc_gbk_paths = [Path(path) for path in read_txt(gbks_filepath)]

    dom_hits = read_dom_hits(dom_hits_filepath)
    domain_colors = read_color_domains_file(domain_colors_filepath)
    detected_motifs = (
        read_detected_motifs(detected_motifs_filepath)
        if detected_motifs_filepath
        else None
    )
    compounds = read_compounds(compounds_filepath) if compounds_filepath else None

    if verbose:
        print("\nVisualising detected motifs...")

    bgc_svgs = dict()
    for bgc_path in bgc_gbk_paths:
        bgc_id = bgc_path.stem

        # header with BGC ID
        svg_text = f"<h1>{bgc_id}</h1>\n"

        # Draw the molecule structure if available
        if compounds:
            svg_text += draw_compounds(compounds.get(bgc_id, []))

        seq_record = list(SeqIO.parse(bgc_path, "genbank"))[0]
        bgc_length = len(seq_record)
        cds_features = [f for f in seq_record.features if f.type == "CDS"]

        # Draw the full BGC
        svg_text += draw_bgc(
            bgc_id=bgc_id,
            bgc_length=bgc_length,
            cds_features=cds_features,
            domain_hits=dom_hits,
            domain_colors=domain_colors,
        )

        # Draw the detected motifs
        for motif_hit in detected_motifs.get(bgc_path.stem, []):
            svg_text += draw_bgc(
                bgc_id=bgc_id,
                bgc_length=bgc_length,
                cds_features=cds_features,
                domain_hits=dom_hits,
                motif_hit=motif_hit,
                domain_colors=domain_colors,
            )

        bgc_svgs[bgc_id] = svg_text
        
    with open(outfile, "w") as f:
        f.write("\n".join(bgc_svgs.values()))
        
    if verbose:
        print(f"  Wrote output to {outfile}")
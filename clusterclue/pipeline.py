from pathlib import Path
from importlib.resources import files

from clusterclue.preprocess.orchestrator import run_preprocess
from clusterclue.detect.motif_detection import main as detect_motifs
from clusterclue.visualize.subcluster_arrower import main as visualize_subclusters


def run_clusterclue(
    out_dir_path,
    motifs_file_path,
    gbks_dir_path,
    existing_clusterfile,
    exclude_name,
    include_contig_edge_clusters,
    max_domain_overlap,
    compounds_filepath,
    cores,
    verbose,
):
    """
    Runs the entire pipeline.
    """
    out_dir_path = Path(out_dir_path)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # Outdated variables
    include_contig_edge_clusters = True

    # Paths to data files
    data_dir = Path(files("clusterclue").joinpath("data"))
    biosynthetic_domains_path = data_dir / "biosynthetic_domains.txt"
    hmm_file_path = data_dir / "biosyn_pfam_112sub.hmm"
    domain_colors_file = data_dir / "domains_color_file.tsv"

    # Step 1: Preprocessing clusters
    preprocess_dir_path = out_dir_path / "preprocess"
    gbks_file = preprocess_dir_path / "input_gbks_paths.txt"
    clusters_file_path = run_preprocess(
        preprocess_dir_path,
        existing_clusterfile,
        gbks_file,
        gbks_dir_path,
        biosynthetic_domains_path,
        hmm_file_path,
        exclude_name,
        include_contig_edge_clusters,
        max_domain_overlap,
        cores,
        verbose,
    )

    # Step 2: Detecting sub-clusters
    detected_motifs = out_dir_path / "detected_motifs.tsv"
    if detected_motifs.is_file():
        if verbose:
            print(
                f"\nSkipping motif detection, because the file already exists: {detected_motifs}"
            )
    else:
        detect_motifs(clusters_file_path, motifs_file_path, detected_motifs)

    # Step 3: Visualizing sub-clusters
    dom_hits_file = preprocess_dir_path / "all_domain_hits.txt"
    out_html = out_dir_path / "detected_motifs.html"

    visualize_subclusters(
        outfile=out_html,
        gbks_filepath=gbks_file,
        dom_hits_filepath=dom_hits_file,
        biosyn_domains_filepath=biosynthetic_domains_path,
        domain_colors_filepath=domain_colors_file,
        detected_motifs_filepath=detected_motifs,
        compounds_filepath=compounds_filepath,
        verbose=verbose,
    )

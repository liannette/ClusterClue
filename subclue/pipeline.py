from pathlib import Path

from subclue.preprocess.orchestrator import run_preprocess
from subclue.detect.gwm_detection import main as detect_motifs
from subclue.visualize.subcluster_arrower import main as visualize_subclusters


def run_subclue(
    out_dir_path,
    motifs_file_path,
    gbks_dir_path,
    existing_clusterfile,
    exclude_name,
    include_contig_edge_clusters,
    hmm_file_path,
    max_domain_overlap,
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

    # Step 1: Preprocessing clusters
    preprocess_dir_path = out_dir_path / "preprocess"
    gbks_file = preprocess_dir_path / "input_gbks_paths.txt"
    domain_hits_file_path = run_preprocess(
        preprocess_dir_path,
        gbks_file,
        gbks_dir_path,
        hmm_file_path,
        exclude_name,
        include_contig_edge_clusters,
        cores,
        verbose,
    )

    # Step 2: Detecting sub-clusters
    detected_motifs = out_dir_path / "detected_motifs.tsv"
    if detected_motifs.is_file():
        if verbose:
            print(f"\nSkipping motif detection, because the file already exists: {detected_motifs}")
    else:
        detect_motifs(domain_hits_file_path, motifs_file_path, detected_motifs, verbose)

    # # Step 3: Visualizing sub-clusters
    # data_dir = Path(__file__).parent.parent / "data"
    # dom_hits_file = preprocess_dir_path / "all_domain_hits.txt"
    # domain_colors_file = data_dir / "domains_color_file.tsv"
    # included_domains = data_dir / "biosynthetic_domains.txt"
    # json_dir = data_dir / "mibig_json_4.0"
    # out_html = out_dir_path / "detected_motifs.html"

    # bgc_path = "input/mibig_gbk_4.0/BGC0002260.gbk"
    # visualize_subclusters(
    #     filenames=gbks_file,
    #     dom_hits_file=dom_hits_file,
    #     include_list=included_domains,
    #     domains_color_file=domain_colors_file,
    #     outfile=out_html,
    #     motif_hits=detected_motifs,
    #     json_dir=json_dir,
    #     verbose=verbose,
    # )

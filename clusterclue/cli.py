from multiprocessing import set_start_method
from multiprocessing import cpu_count, Queue, Process
import sys
import argparse
import logging
import time
from pathlib import Path
from clusterclue.pipeline import create_new_motifs, detect_existing_motifs
from clusterclue.utils import listener_process, worker_configurer

# Python 3.12 issues a DeprecationWarning when using os.fork() in a multi-threaded process,
# because forking with active threads can cause deadlocks due to lock states not being 
# safely inherited by the child process. To avoid this, we explicitly set the 
# multiprocessing start method to 'spawn' which creates a fresh Python interpreter process
set_start_method("spawn", force=True)

def get_commands():
    parser = argparse.ArgumentParser(
        description="",
    )
    subparsers = parser.add_subparsers(
        dest="mode",
        required=True
    )

    build = subparsers.add_parser(
        "build",
        help="Generate new subcluster motifs from input gene clusters.",
    )
    build.add_argument(
        "--out",
        dest="out_dir_path",
        required=True,
        metavar="<dir>",
        help="Output directory, this will contain all output data files.",
    )
    build.add_argument(
        "--gbks",
        dest="gbks_dir_path",
        required=True,
        metavar="<dir>",
        help="Input directory containing gbk files of the gene clusters.",
    )
    build.add_argument(
        "--hmm",
        dest="hmm_file_path",
        required=True,
        metavar="<file>",
        help="Path to the HMM file containing protein domain HMMs that has been "
        "processed with hmmpress.",
    )
    build.add_argument(
        "--max_domain_overlap",
        dest="max_domain_overlap",
        default=0.1,
        metavar="<float>",
        help="Specify at which overlap percentage domains are considered to overlap. "
        "Domain with the best score is kept (default=0.1).",
    )
    build.add_argument(
        "-c",
        "--cores",
        dest="cores",
        default=cpu_count(),
        help="Set the number of cores the script may use (default: use all "
        "available cores)",
        type=int,
        metavar="<int>",
    )
    build.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="Prints more detailed information.",
    )
    build.add_argument(
        "--incl_contig_edge",
        dest="include_contig_edge_clusters",
        action="store_true",
        default=False,
        help="Include clusters that lie on a contig edge. (default = false)",
    )
    build.add_argument(
        "--min_genes_per_bgc",
        dest="min_genes_per_bgc",
        default=2,
        type=int,
        metavar="<int>",
        help="Minimum number of non-empty genes required in a cluster to "
        "be included in the analysis. A gene is considered empty if it "
        "lacks any protein domains (default: 2).",
    )
    build.add_argument(
        "--similarity_filtering",
        action="store_true",
        default=False,
        help="If provided, similarity filtering of the tokenized clusters "
        "will be performed",
    )
    build.add_argument(
        "--similarity_cutoff",
        dest="similarity_cutoff",
        default=0.95,
        type=float,
        metavar="<float>",
        help="Cutoff for cluster similarity in redundancy filtering. It refers "
        "to the adjecency index of domains (default:0.95).",
    )
    build.add_argument(
        "--remove_infrequent_genes",
        action="store_true",
        default=False,
        help="If provided, genes will be removed from the bgcs if they occur "
        "less than --min_gene_occurrence times in the dataset.",
    )
    build.add_argument(
        "--min_gene_occurrence",
        default=3,
        type=int,
        metavar="<int>",
        help="Remove tokenized genes that occur fewer than the specified "
        "number of times in the data (default: 3).",
    )
    # PRESTO-STAT
    build.add_argument(
        "--no_presto_stat",
        action="store_false",
        dest="run_stat",
        default=True,
        help="If provided, PRESTO-STAT will not be run",
    )
    build.add_argument(
        "--stat_modules",
        dest="stat_modules_file_path",
        default=None,
        metavar="<file>",
        help="Text file containing previously inferred subclusters to detect "
        "in the input. If not provided, PRESTO-STAT will run to detect new "
        "subclusters in the input (default: None)",
    )
    build.add_argument(
        "--stat_pval_cutoff",
        dest="stat_pval_cutoff",
        default=0.1,
        type=float,
        help="P-value cutoff for determining a significant interaction in module detection "
        "(default: 0.1)",
        metavar="<float>",
    )
    build.add_argument(
        "--stat_families",
        dest="stat_n_families_range",
        nargs="+",
        type=int,
        default=[],
        metavar="<int>",
        help="Specify one or more integers to define the number of families for "
        "clustering STAT modules. Each integer represents a different clustering "
        "configuration (default: off).",
    )
    # PRESTO-TOP\
    build.add_argument(
        "--no_presto_top",
        dest="run_top",
        action="store_false",
        default=True,
        help="If provided, PRESTO-TOP will not be run",
    )
    build.add_argument(
        '--top_model',
        dest="top_model_file_path",
        default=None,
        metavar="<file>",
        help='Use PRESTO-TOP with existing sub-cluster motifs in an LDA model. '
             'Supply here the path to the model. In that location there should be also '
             'model.dict, model.expElogbeta.npy, model.id2word, model.state, '
             'model.state.sstats.npy',
    )
    build.add_argument(
        "--top_topics",
        dest="top_n_topics",
        help="Amount of topics to use for the LDA model in PRESTO-TOP (default: 1000)",
        default=1000,
        type=int,
        metavar="<int>"
    )
    build.add_argument(
        "--top_amplify",
        help="Amplify the dataset in order to achieve a better LDA model. Each BGC will be present "
             "amplify times in the dataset. After calculating the LDA model the dataset will be "
             "scaled back to normal.",
        type=int,
        default=None,
        metavar="<int>"
    )
    build.add_argument(
        "--top_iterations",
        help="Amount of iterations for training the LDA model (default: 500)",
        default=500,
        type=int,
        metavar="<int>"
    )
    build.add_argument(
        "--top_chunksize",
        default=2000,
        type=int,
        help='The chunksize used to train the model (default: 2000)',
        metavar="<int>"
    )
    build.add_argument(
        "--top_update",
        help="If provided and a model already exists, the existing model will be updated with "
             "original parameters, new parameters cannot be passed in the LdaMulticore version.",
        default=False,
        action="store_true"
    )
    build.add_argument(
        "--top_visualise",
        help="Make a visualation of the LDA model with pyLDAvis (html file). If number of topics "
             "is too big this might fail. No visualisation will then be made",
        default=False,
        action="store_true"
    )
    build.add_argument(
        "--top_alpha",
        default="symmetric",
        help="alpha parameter for the LDA model, see gensim. Options: (a)symmetric, auto, or <int>"
    )
    build.add_argument(
        "--top_beta",
        default="symmetric",
        help="beta parameter for the LDA model, see gensim. Options: (a)symmetric, auto, or <int>"
    )
    build.add_argument(
        "--top_plot",
        help="If provided: make plots about several aspects of the presto-top output",
        default=False,
        action="store_true"
    )
    build.add_argument(
        "--top_feat_num",
        help="Include the first feat_num features for each topic (default: 75)",
        type=int,
        default=75,
        metavar="<int>"
    )
    build.add_argument(
        "--top_min_feat_score",
        help="Only include features until their scores add up to this number (default: 0.95) Can "
             "be combined with feat_num, where feat_num features are selected or features that "
             "add up to min_feat_score",
        type=float,
        default=0.95,
        metavar="<float>"
    )
    # GWMs
    build.add_argument(
        "--k_values",
        nargs="+",
        type=int,
        default=[1500,],
        metavar="<int>",
        help="Specify one or more integers to define the number of motifs (GWMS). "
        "Each integer represents a different clustering configuration (default: 1500).",
    )
    build.add_argument(
        "--ref_sc",
        dest="reference_subclusters_filepath",
        default=None,
        metavar="<file>",
        help="TSV file with annotated subclusters. "
        "Used to select best hyperparameters for motif generation."
    )
    build.add_argument(
        "--ref_gbks",
        dest="reference_gbks_dirpath",
        default=None,
        metavar="<dir>",
        help="Input directory containing gbk files of gene clusters with annotated subclusters. "
        "Used to select best hyperparameters for motif generation.",
    )
    build.add_argument(
        "--visualise_evaluation",
        dest="visualise_evaluation",
        action="store_true",
        default=False,
        help="Visualize the detected motifs in the reference clusters in html reports.",
    )
    build.add_argument(
        "--compound_smiles",
        dest="compound_smiles_filepath",
        default=None,
        metavar="<file>",
        help="Path to a TSV file containing bgc_id, compound_name, compound_smiles. "
        "If provided, compound structures will be visualized in the html reports.",
    )

    detect = subparsers.add_parser(
        "detect",
        help="Detect subcluster motifs in input gene clusters using existing motifs.",
    )
    detect.add_argument(
        "--gwms",
        dest="gwms_filepath",
        required=True,
        metavar="<dir>",
        help="Input directory containing gene weight matrices (GWMs) of subcluster motifs.",
    )
    detect.add_argument(
        "--out",
        dest="out_dir_path",
        required=True,
        metavar="<dir>",
        help="Output directory, this will contain all output data files.",
    )
    detect.add_argument(
        "--gbks",
        dest="gbk_dir_path",
        required=True,
        metavar="<dir>",
        help="Input directory containing gbk files of the gene clusters.",
    )
    detect.add_argument(
        "--hmm",
        dest="hmm_file_path",
        required=True,
        metavar="<file>",
        help="Path to the HMM file containing protein domain HMMs that has been "
        "processed with hmmpress.",
    )
    detect.add_argument(
        "--max_domain_overlap",
        dest="max_domain_overlap",
        default=0.1,
        metavar="<float>",
        help="Specify at which overlap percentage domains are considered to overlap. "
        "Domain with the best score is kept (default=0.1).",
    )
    detect.add_argument(
        "--visualize_hits",
        dest="visualize_hits",
        action="store_true",
        default=False,
        help="Visualize detected motif hits.",
    )
    detect.add_argument(
        "--compound_smiles",
        dest="compound_smiles_filepath",
        default=None,
        metavar="<file>",
        help="Path to a TSV file containing bgc_id, compound_name, compound_smiles. "
        "If provided, compound structures will be visualized in the html reports.",
    )
    detect.add_argument(
        "-c",
        "--cores",
        dest="cores",
        default=cpu_count(),
        help="Set the number of cores the script may use (default: use all "
        "available cores)",
        type=int,
        metavar="<int>",
    )
    detect.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="Prints more detailed information.",
    )
    args = parser.parse_args()

    return args


def main():
    """
    Main function to execute the iPRESTO pipeline.

    This function retrieves command line arguments, prints them if verbose mode is enabled,
    and then runs the main pipeline with the provided arguments.

    Parameters:
    None

    Returns:
    None
    """
    start_time = time.time()

    cmd = get_commands()

    Path(cmd.out_dir_path).mkdir(parents=True, exist_ok=True)
    log_file_path = Path(cmd.out_dir_path) / "clusterclue.log"

    # Set up multiprocessing-friendly logging
    queue = Queue(-1)
    listener = Process(target=listener_process, args=(queue, log_file_path, cmd.verbose))
    listener.start()
    worker_configurer(queue)
    logger = logging.getLogger("clusterclue.cli")
    logger.info("Command: %s", " ".join(sys.argv))

    if cmd.mode == "build":
        create_new_motifs(cmd, queue)

    elif cmd.mode == "detect":
        detect_existing_motifs(cmd, queue)

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info("Total runtime: %d hours and %d minutes", int(hours), int(minutes))

    queue.put(None)
    listener.join()
    queue.close()
    queue.join_thread()


if __name__ == "__main__":
    main()

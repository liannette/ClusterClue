import argparse
import sys

from clusterclue.pipeline import run_clusterclue


def get_commands():
    parser = argparse.ArgumentParser()
    # Output directory
    parser.add_argument(
        "--out",
        dest="out_dir_path",
        required=True,
        metavar="<dir>",
        help="Output directory, this will contain all output data files.",
    )
    # Input options
    parser.add_argument(
        "--motifs",
        dest="motifs_file_path",
        required=True,
        metavar="<file>",
        help="Path to the motifs file containing subcluster weights and thresholds.",
    )
    parser.add_argument(
        "--gbks",
        dest="gbk_dir_path",
        metavar="<dir>",
        help="Input directory containing gbk files of the gene clusters.",
    )
    parser.add_argument(
        "--clusters",
        dest="existing_clusterfile",
        default=None,
        metavar="<file>",
        help="Path to a previously created CSV file of tokenized clusters. "
        "This option overrides the --gbks argument, although --gbks must "
        "still be provided.",
    )
    # Preprocessing settings
    parser.add_argument(
        "--hmm",
        dest="hmm_file_path",
        metavar="<file>",
        help="Path to the HMM file containing protein domain HMMs that has been "
        "processed with hmmpress.",
    )
    parser.add_argument(
        "--exclude_name",
        dest="exclude_name",
        default=["final"],
        nargs="+",
        metavar="<str>",
        help="If any string in this list occurs in the gbk filename, this "
        "file will not be used for the analysis (default: ['final']).",
    )
    parser.add_argument(
        "--incl_contig_edge",
        dest="include_contig_edge_clusters",
        action="store_true",
        default=False,
        help="Include clusters that lie on a contig edge. (default = false)",
    )
    parser.add_argument(
        "--max_domain_overlap",
        dest="max_domain_overlap",
        default=0.1,
        metavar="<float>",
        help="Specify at which overlap percentage domains are considered to overlap. "
        "Domain with the best score is kept (default=0.1).",
    )
    # Other arguments
    parser.add_argument(
        "-c",
        "--cores",
        dest="cores",
        default=1,
        help="Set the number of cores the script may use (default: 1)",
        type=int,
        metavar="<int>",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="Prints more detailed information.",
    )

    args = parser.parse_args()

    # Validation logic
    if not args.existing_clusterfile:
        if not args.gbk_dir_path or not args.hmm_file_path:
            parser.error("Either --gbks and --hmm must be provided, or --clusters must be specified.")

    return args


def main():
    """
    Main function to execute ClusterClue.

    This function retrieves command line arguments, prints them if verbose mode is enabled,
    and then runs the main pipeline with the provided arguments.

    Parameters:
    None

    Returns:
    None
    """
    # Get the command line arguments
    cmd = get_commands()

    # Print the command line arguments if verbose
    if cmd.verbose:
        print("Command:", " ".join(sys.argv))
        print(cmd)

    # Execute the main pipeline with provided arguments
    run_clusterclue(
        cmd.out_dir_path,
        cmd.motifs_file_path,
        cmd.gbk_dir_path,
        cmd.existing_clusterfile,
        cmd.exclude_name,
        cmd.include_contig_edge_clusters,
        cmd.hmm_file_path,
        cmd.max_domain_overlap,
        cmd.cores,
        cmd.verbose,
    )


if __name__ == "__main__":
    main()

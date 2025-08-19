import os
from pathlib import Path

from subclue.preprocess.tokenize.process_domtable import process_domtables
from subclue.preprocess.tokenize.process_fasta import process_fastas
from subclue.preprocess.tokenize.process_gbk import process_gbks, write_gbk_paths_file


class TokenizeOrchestrator:
    def run(
        self,
        all_domains_file,
        gbks_file,
        gbks_dir_path,
        hmm_file_path,
        exclude_name,
        include_contig_edge_clusters,
        cores,
        verbose,
    ):
        """Wrapper for tokenization of clusters.

        This function tokenizes clusters by processing gbk files into fasta
        files, running hmmscan on the fastas to generate domtables, parsing
        the domtables into tokenized clusters, and writing the tokenized
        clusters to a file.

        Args:
            all_domains_file (str): Path to the output file where all domain hits will be saved.
            gbks_file (str): Path to a file containing paths to the input GenBank files.
            gbks_dir_path (str): Path to the folder containing gbk files.
            hmm_file_path (str): Path to the HMM file to be used as the database.
            exclude_name (list of str): List of substrings; files will be excluded if part of the file name is present in this list.
            include_contig_edge_clusters (bool): Whether to include clusters on contig edges.
            max_domain_overlap (float): If two domains overlap more than this value, only the domain with the highest score is kept.
            cores (int): Number of CPU cores to use for parallel processing.
            verbose (bool): If True, print additional info to stdout.

        Returns:
            str: Path to the tokenized clusters file.
        """
        outdir_path = Path(all_domains_file).parent

        # Step 1: Processing gbk files into fasta files
        fastas_dir_path = outdir_path / "fastas"
        os.makedirs(fastas_dir_path, exist_ok=True)
        process_gbks(
            gbks_dir_path,
            gbks_file,
            fastas_dir_path,
            exclude_name,
            include_contig_edge_clusters,
            cores,
            verbose,
        )

        # Step 2: Processing fastas with hmmscan to generate domtables
        domtables_dir_path = outdir_path / "domtables"
        os.makedirs(domtables_dir_path, exist_ok=True)
        process_fastas(
            fastas_dir_path,
            domtables_dir_path,
            hmm_file_path,
            cores,
            verbose,
        )

        # Step 3: Processing domtables
        process_domtables(
            domtables_dir_path,
            all_domains_file,
            cores,
            verbose,
        )

        # Print paths
        if verbose:
            print(f"Domain hits has been saved to {all_domains_file}")

        return all_domains_file

from pathlib import Path

from ipresto.preprocess.tokenize.process_domtable import process_domtables
from ipresto.preprocess.tokenize.process_fasta import process_fastas
from ipresto.preprocess.tokenize.process_gbk import process_gbks, write_gbk_paths_file


class TokenizeOrchestrator:
    def run(
        self,
        clusters_file_path,
        gene_counts_file_path,
        gbks_dir_path,
        hmm_file_path,
        exclude_name,
        include_contig_edge_clusters,
        min_genes,
        max_domain_overlap,
        cores,
        verbose,
    ):
        """Wrapper for tokenization of clusters.

        This function tokenizes clusters by processing gbk files into fasta
        files, running hmmscan on the fastas to generate domtables, parsing
        the domtables into tokenized clusters, and writing the tokenized
        clusters to a file.

        Args:
            clusters_file_path (str): Path to the output file where tokenized clusters will be saved.
            gbks_dir_path (str): Path to the folder containing gbk files.
            hmm_file_path (str): Path to the HMM file to be used as the database.
            exclude_name (list of str): List of substrings; files will be excluded if part of the
                file name is present in this list.
            include_contig_edge_clusters (bool): Whether to include clusters on contig edges.
            min_genes (int): Minimum number of genes required for processing.
            max_domain_overlap (float): If two domains overlap more than this
                value, only the domain with the highest score is kept.
            cores (int): Number of CPU cores to use for parallel processing.
            verbose (bool): If True, print additional info to stdout.

        Returns:
            str: Path to the tokenized clusters file.
        """
        if verbose:
            print("\nTokenizing the BGC genes into protein domain combinations")
            
        outdir_path = Path(clusters_file_path).parent

        # write the paths of all input gbks to a file
        gbks_file = outdir_path / "input_gbks_paths.txt"
        if not gbks_file.exists():
            write_gbk_paths_file(gbks_dir_path, gbks_file)

        # Step 1: Processing gbk files into fasta files
        fastas_dir_path = outdir_path / "fastas"
        fastas_dir_path.mkdir(exist_ok=True, parents=True)
        process_gbks(
            gbks_dir_path,
            fastas_dir_path,
            exclude_name,
            include_contig_edge_clusters,
            cores,
            verbose,
        )

        # Step 2: Processing fastas with hmmscan to generate domtables
        domtables_dir_path = outdir_path / "domtables"
        domtables_dir_path.mkdir(exist_ok=True, parents=True)
        process_fastas(
            fastas_dir_path,
            domtables_dir_path,
            hmm_file_path,
            cores,
            verbose,
        )

        # Step 3: Processing domtables into tokenized clusters
        domain_hits_file_path = outdir_path / "all_domain_hits.txt"
        process_domtables(
            domtables_dir_path,
            clusters_file_path,
            gene_counts_file_path,
            domain_hits_file_path,
            min_genes,
            max_domain_overlap,
            cores,
            verbose,
        )

        # Print paths
        if verbose:
            print("\nTokenization complete.")
            print(f"Tokenized clusters have been saved to {clusters_file_path}")
            print(f"Gene counts have been saved to {gene_counts_file_path}")
            print(f"Summary of domain hits has been saved to {domain_hits_file_path}")

        return clusters_file_path

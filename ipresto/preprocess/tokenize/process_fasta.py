from multiprocessing import Pool
from functools import partial
from pathlib import Path
from pyhmmer.hmmer import hmmscan
from pyhmmer.plan7 import HMMFile
from pyhmmer.easel import SequenceFile


def run_hmmscan(fasta_file_path, hmm_file_path, out_folder, verbose):
    """Runs pyhmmer hmmscan and saves the output to a specified folder.

    If the output file already exists in the specified folder, it will be reused.

    Args:
        fasta_file_path (str): Path to the input FASTA file.
        hmm_file_path (str): Path to the HMM file to be used as the database.
        out_folder (str): Directory where the output file will be saved.
        verbose (bool): If True, prints the hmmscan command being executed.
    Returns:
        str: Status of the operation, either "existed", "failed" or "converted".
    """
    fasta_file_path = Path(fasta_file_path)
    name = fasta_file_path.stem
    out_file_path = Path(out_folder) / f"{name}.domtable"

    # check if the domtable file already exists in the out dir
    if out_file_path.is_file():
        return "existed"

    # run hmmscan
    try:
        with HMMFile(hmm_file_path) as hmm_file, \
             SequenceFile(fasta_file_path, digital=True) as seq_file:
            hits = list(hmmscan(seq_file, hmm_file, cpus=1, bit_cutoffs="trusted"))
    except Exception as e:
        if verbose:
            print(f"Error running hmmscan on {fasta_file_path}: {e}")
        return "failed"

    # Write the hits to the output file
    with open(out_file_path, "wb") as out_file:
        # Write header only once
        hits[0].write(out_file, format="domains", header=True)
        # Write the remaining hits without repeating the header
        for hit in hits[1:]:
            hit.write(out_file, format="domains", header=False)
    return "converted"


def process_fastas(fasta_dir_path, domtables_dir_path, hmm_file_path, cores, verbose):
    """Runs hmmscan on all provided fasta files using the specified HMM file as the database.

    Args:
        fasta_file_paths (list): List of paths to the input FASTA files.
        domtables_dir_path (str): Path to the directory where output domtables will be stored.
        hmm_file_path (str): Path to the HMM file to be used as the database.
        verbose (bool): If True, print additional information during execution.
        cores (int): Number of CPU cores to use for parallel processing.

    Returns:
        list: List of paths to the generated domtable files.
    """
    if verbose:
        print("\nRunning hmmscan on fastas to generate domtables...")

    fasta_file_paths = list(Path(fasta_dir_path).glob("*.fasta"))

    # Process each fasta file in parallel
    with Pool(cores, maxtasksperchild=100) as pool:
        process_func = partial(
            run_hmmscan,
            hmm_file_path=hmm_file_path,
            out_folder=domtables_dir_path,
            verbose=verbose,
        )
        results = pool.map(process_func, fasta_file_paths)

    # Print summary of processing
    if verbose:
        status_counts = {
            status: results.count(status) for status in ["converted", "existed", "failed"]
        }
        n_converted = status_counts["converted"]
        n_existed = status_counts["existed"]
        n_failed = status_counts["failed"]

        print(f"\nProcessed {len(fasta_file_paths)} fasta files:")
        if n_failed > 0:
            print(f" - {n_failed} fasta files failed to be converted into domtables")
        if n_existed > 0:
            print(f" - {n_existed} domtables already existed in the output folder")
        if n_converted > 0:
            print(f" - {n_converted} fasta files were converted into domtables")

from multiprocessing import Pool
from functools import partial
from pathlib import Path
from pyhmmer.hmmer import hmmscan
from pyhmmer.plan7 import HMMFile
from pyhmmer.easel import SequenceFile


def run_hmmscan(fasta_file_path, hmm_file_path, out_file_path):
    """Runs pyhmmer hmmscan and saves the output to a specified file.

    Args:
        fasta_file_path (str): Path to the input FASTA file.
        hmm_file_path (str): Path to the HMM file to be used as the database.
        out_file_path (str): Path to the output file where the results will be saved.

    Returns:
        None
    """
    # run hmmscan
    with HMMFile(hmm_file_path) as hmm_file, \
            SequenceFile(fasta_file_path, digital=True) as seq_file:
        hits = list(hmmscan(seq_file, hmm_file, cpus=1, bit_cutoffs="trusted"))

    # Write the hits to the output file
    with open(out_file_path, "wb") as out_file:
        # Write header only once
        hits[0].write(out_file, format="domains", header=True)
        # Write the remaining hits without repeating the header
        for hit in hits[1:]:
            hit.write(out_file, format="domains", header=False)


def run_hmmscan_wrapper(fasta_file_path, hmm_file_path, out_folder, verbose):
    fasta_file_path = Path(fasta_file_path)
    out_file_path = Path(out_folder) / f"{fasta_file_path.stem}.domtable"

    # check if the domtable file already exists in the out dir
    if out_file_path.is_file():
        return "existed"
    # run hmmscan
    try:
        run_hmmscan(fasta_file_path, hmm_file_path, out_file_path)
        return "converted"
    # handle errors
    except Exception as e:
        if verbose:
            print(f"  Unexpected error processing {fasta_file_path.name}: {e}")
        return "failed"


def process_fastas(fasta_dir_path, domtables_dir_path, hmm_file_path, cores, verbose):
    """Runs hmmscan on all provided fasta files using the specified HMM file as the database.

    Args:
        fasta_file_paths (list): List of paths to the input FASTA files.
        domtables_dir_path (str): Path to the directory where output domtables will be stored.
        hmm_file_path (str): Path to the HMM file to be used as the database.
        cores (int): Number of CPU cores to use for parallel processing.
        verbose (bool): If True, print additional information during execution.

    Returns:
        None
    """
    if verbose:
        print("\nRunning hmmscan on fastas to generate domtables...")

    fasta_file_paths = list(Path(fasta_dir_path).glob("*.fasta"))

    # Process each fasta file in parallel
    with Pool(cores, maxtasksperchild=100) as pool:
        process_func = partial(
            run_hmmscan_wrapper,
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

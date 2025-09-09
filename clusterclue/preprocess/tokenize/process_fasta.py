import os
import subprocess
from glob import iglob
from multiprocessing import Pool


def run_hmmscan(fasta_file_path, hmm_file_path, out_folder, verbose):
    """Runs the hmmscan tool on a given FASTA file using a specified HMM file, and saves the output to a specified folder.
    If the output file already exists in the specified folder or in an existing domain folder, it will be reused.

    Args:
        fasta_file_path (str): Path to the input FASTA file.
        hmm_file_path (str): Path to the HMM file.
        out_folder (str): Directory where the output file will be saved.
        verbose (bool): If True, prints the hmmscan command being executed.
    Returns:
        str: Status of the operation, either "copied", "existed", or "converted".
    Raises:
        SystemExit: If the input FASTA file does not exist.
    """

    name = os.path.split(fasta_file_path)[1].split(".fasta")[0]
    out_file_name = name + ".domtable"
    out_file_path = os.path.join(out_folder, out_file_name)

    if not os.path.isfile(fasta_file_path):
        print(f"Error running hmmscan: {fasta_file_path} doesn't exist")
        return "failed"

    # check if the domtable file already exists in the out dir
    if os.path.isfile(out_file_path):
        return "existed"

    # run hmmscan
    log_file_path = os.devnull
    cmd = f"hmmscan -o {log_file_path} --cpu 0 --domtblout {out_file_path} --cut_tc {hmm_file_path} {fasta_file_path}"
    subprocess.check_call(cmd, shell=True)
    return "converted"


def process_fastas(fasta_dir_path, domtables_dir_path, hmm_file, cores, verbose):
    """Runs hmmscan on all provided fasta files using the specified HMM file as the database.

    Args:
        fasta_file_paths (list): List of paths to the input FASTA files.
        domtables_dir_path (str): Path to the directory where output domtables will be stored.
        hmm_file (str): Path to the HMM file to be used as the database.
        verbose (bool): If True, print additional information during execution.
        cores (int): Number of CPU cores to use for parallel processing.

    Returns:
        list: List of paths to the generated domtable files.
    """
    if verbose:
        print("\nRunning hmmscan on fastas to generate domtables...")

    fasta_file_paths = list(iglob(os.path.join(fasta_dir_path, "*.fasta")))

    # Process each fasta file in parallel
    done = []
    pool = Pool(
        cores, maxtasksperchild=100
    )  # maxtasksperchild=1:each process respawns after completing 1 chunk
    for file_path in fasta_file_paths:
        pool.apply_async(
            run_hmmscan,
            args=(file_path, hmm_file, domtables_dir_path, verbose),
            callback=lambda x: done.append(x),
            error_callback=lambda e: print(f"Error processing {file_path}: {e}"),
        )
    pool.close()
    pool.join()  # make the code in main wait for the pool processes to finish

    # Print summary of processing
    if verbose:
        status_counts = {
            status: done.count(status) for status in ["converted", "existed", "failed"]
        }
        n_converted = status_counts["converted"]
        n_existed = status_counts["existed"]
        n_failed = status_counts["failed"]

        print(f"\nProcessed {len(fasta_file_paths)} fasta files:")
        print(f" - {n_existed} domtables already existed in the output folder")
        print(f" - {n_converted} fasta files were converted into domtables")
        print(f" - {n_failed} fasta files failed be converted into domtables")

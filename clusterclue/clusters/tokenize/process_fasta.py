import logging
from multiprocessing import Pool
from functools import partial
from pathlib import Path
from pyhmmer.hmmer import hmmscan
from pyhmmer.plan7 import HMMFile
from pyhmmer.easel import SequenceFile, Alphabet
from clusterclue.utils import worker_init

logger = logging.getLogger(__name__)


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
            SequenceFile(fasta_file_path, digital=True, alphabet=Alphabet.amino()) as seq_file:
        hits = list(hmmscan(seq_file, hmm_file, cpus=1, bit_cutoffs="trusted"))

    # Write the hits to the output file
    if not hits:
        print(f"Warning: No hits found. Creating empty output file: {out_file_path}")
        with open(out_file_path, "w") as out_file:
            out_file.write("# No hits found\n")
        return
    
    with open(out_file_path, "wb") as out_file:
        # Write header with first hit
        hits[0].write(out_file, format="domains", header=True)
        
        # Write remaining hits without repeating the header
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
        logger.error(f"Unexpected error processing {fasta_file_path.name}: {e}")
        return "failed"


def process_fastas(fasta_dir_path, domtables_dir_path, hmm_file_path, cores, verbose, log_queue):
    """Runs hmmscan on all provided fasta files using the specified HMM file as the database.

    Args:
        fasta_file_paths (list): List of paths to the input FASTA files.
        domtables_dir_path (str): Path to the directory where output domtables will be stored.
        hmm_file_path (str): Path to the HMM file to be used as the database.
        cores (int): Number of CPU cores to use for parallel processing.
        verbose (bool): If True, print additional information during execution.
        log_queue (multiprocessing.Queue): Queue for logging in multiprocessing.

    Returns:
        None
    """
    logger.info("Processing fastas with hmmscan to generate domtables...")

    fasta_file_paths = sorted(Path(fasta_dir_path).glob("*.fasta"))

    # Process each fasta file in parallel
    pool = Pool(
        cores,
        maxtasksperchild=100,
        initializer=worker_init,
        initargs=(log_queue,)
    )
    with pool:
        process_func = partial(
            run_hmmscan_wrapper,
            hmm_file_path=hmm_file_path,
            out_folder=domtables_dir_path,
            verbose=verbose,
        )
        results = pool.map(process_func, fasta_file_paths)

    # Print summary of processing
    status_counts = {
        status: results.count(status) for status in ["converted", "existed", "failed"]
    }
    summary_parts = []
    if status_counts["converted"] > 0:
        summary_parts.append(f"{status_counts['converted']} converted")
    if status_counts["existed"] > 0:
        summary_parts.append(f"{status_counts['existed']} skipped (already existed)")
    if status_counts["failed"] > 0:
        summary_parts.append(f"{status_counts['failed']} failed")
    summary = ', '.join(summary_parts) if summary_parts else "no files processed"
    logger.info(f"FASTA to domtable: {len(fasta_file_paths)} total - {summary}")

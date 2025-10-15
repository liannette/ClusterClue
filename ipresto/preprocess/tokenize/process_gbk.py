import logging
from collections import OrderedDict
from multiprocessing import Pool
from functools import partial
from Bio import SeqIO
from pathlib import Path

logger = logging.getLogger(__name__)


def write_gbk_paths_file(gbks_dir_path, out_file_path):
    # write names of all input gbk to file
    gbk_file_paths = list(Path(gbks_dir_path).glob("*.gbk"))
    with open(out_file_path, "w") as f:
        for gbk in gbk_file_paths:
            f.write(f"{gbk}\n")
                    

def convert_gbk2fasta(
    gbk_file_path,
    out_file_path,
    include_contig_edge_clusters,
    verbose,
):
    """Convert a GenBank (gbk) file to a FASTA file.

    Parameters:
    gbk_file_path (str): Path to the input gbk file.
    out_file_path (str): Path where the output FASTA file will be saved.
    include_contig_edge_clusters (bool): If True, include clusters at contig edges.
    verbose (bool): If True, print additional information to stdout.

    Returns:
        str:
            - "filtered" if the file is excluded due to contig edge
            - "converted" if the conversion to FASTA is successful.
    """
    bgc_name = Path(gbk_file_path).stem

    # parse the gbk file for conversion to fasta
    record = list(SeqIO.read(gbk_file_path, "genbank"))

    # check if contig edge cluster
    if not include_contig_edge_clusters:
        for feature in record.features:
            if feature.type == "protocluster":
                contig_edge = feature.qualifiers.get("contig_edge")[0]
                if contig_edge == "True":
                    logger.debug(f"  excluding {bgc_name}: contig edge")
                    return "filtered"
                break

    # extract sequences and write to fasta
    seqs = OrderedDict()
    num_genes = 0
    for feature in record.features:
        if feature.type == "CDS":
            gene_id = "gid:"
            if "gene" in feature.qualifiers:
                gene_id += feature.qualifiers.get("gene", "")[0]
                gene_id = gene_id.replace("_", "-")
            protein_id = "pid:"
            if "protein_id" in feature.qualifiers:
                protein_id += feature.qualifiers.get("protein_id", "")[0]
                protein_id = protein_id.replace("_", "-")
            start = feature.location.start
            end = feature.location.end
            strand = feature.location.strand
            if strand == 1:
                strand = "+"
            else:
                strand = "-"
            loc = "loc:{};{};{}".format(start, end, strand)
            head = "_".join([bgc_name, gene_id, protein_id, loc])
            head = head.replace(">", "")  # loc might contain this
            head = head.replace("<", "")
            header = ">{}_{}".format(head, num_genes + 1)
            header = header.replace(" ", "")  # hmmscan uses space as delim
            seqs[header] = feature.qualifiers.get("translation", [""])[0]
            if seqs[header] == "":
                raise ValueError(f"{gene_id} does not have a translation")
            num_genes += 1

    # write the fasta file
    with open(out_file_path, "w") as out:
        for seq in seqs:
            compl_header = "{}/{}".format(seq, num_genes)
            out.write("{}\n{}\n".format(compl_header, seqs[seq]))
    return "converted"


def convert_gbk2fasta_wrapper(
    gbk_file_path,
    out_folder,
    include_contig_edge_clusters,
    exclude_name,
    verbose,
):
    """Convert a GenBank (gbk) file to a FASTA file.

    Parameters:
        gbk_file_path (str): Path to the input gbk file.
        out_folder (str): Directory where the output FASTA file will be saved.
        include_contig_edge_clusters (bool): If True, include clusters at contig edges.
        exclude_name (list): List of words; files containing any of these words in their name will be excluded.
        verbose (bool): If True, print additional information to stdout.

    Returns:
        str:
            - "excluded" if the file name contains any word from the exclude list.
            - "existed" if the FASTA file already exists in the output folder.
            - "failed" if there is an error parsing the gbk file.
            - "filtered" if the file is excluded due to contig edge
            - "converted" if the conversion to FASTA is successful.
    """
    gbk_file_path = Path(gbk_file_path)
    out_file_path = Path(out_folder) / f"{gbk_file_path.stem}.fasta"

    # exclude files with certain words in the name
    if any([word in str(gbk_file_path.stem) for word in exclude_name]):
        return "excluded"
    # check if the fasta file already exists
    if out_file_path.exists():
        return "existed"
    # convert gbk to fasta
    try:
        status = convert_gbk2fasta_wrapper(
            gbk_file_path,
            out_folder,
            include_contig_edge_clusters,
            verbose,
        )
        return status
    # handle errors
    except Exception as e:
        logger.error(f"  Unexpected error processing {gbk_file_path.name}: {e}")
        return "failed"


def process_gbks(
    gbks_dir_path,
    fastas_dir_path,
    exclude_name,
    include_contig_edge_clusters,
    cores,
    verbose,
):
    """Convert gbk files from input folder to fasta files for each gbk file.

    Args:
        gbks_dir_path (str): Path to the folder containing gbk files.
        fastas_dir_path (str): Path to the folder where fasta files will be saved.
        exclude_name (list of str): List of substrings; files will be excluded if 
            part of the file name is present in this list.
        include_contig_edge_clusters (bool): Whether to include contig edges.
        cores (int): Number of CPU cores to use for parallel processing.
        verbose (bool): If True, print additional info to stdout.

    Returns:
        fastas_file_paths (list of str): List of all fasta files.
    """
    logger.info("Processing gbk files into fasta files...")

    gbk_file_paths = list(Path(gbks_dir_path).glob("*.gbk"))

    # Remove fasta files of bgcs that did not have a gbk file
    cluster_ids = [fp.stem for fp in gbk_file_paths]
    for fasta_file_path in Path(fastas_dir_path).glob("*.fasta"):
        if fasta_file_path.stem not in cluster_ids:
            fasta_file_path.unlink()

    # Process each gbk file in parallel
    with Pool(cores, maxtasksperchild=100) as pool:
        process_func = partial(
            convert_gbk2fasta,
            out_folder=fastas_dir_path,
            include_contig_edge_clusters=include_contig_edge_clusters,
            exclude_name=exclude_name,
            verbose=verbose,
        )
        results = pool.map(process_func, gbk_file_paths)
        
    # Print summary of processing
    status_counts = {
        status: results.count(status)
        for status in ["converted", "existed", "excluded", "failed", "filtered"]
    }
    n_converted = status_counts["converted"]
    n_existed = status_counts["existed"]
    n_excluded = status_counts["excluded"]
    n_failed = status_counts["failed"]
    n_filtered = status_counts["filtered"]

    logger.info(f"Processed {len(gbk_file_paths)} gbk files:")
    if n_existed > 0:
        logger.info(f" - {n_existed} fasta files already existed in the output folder")
    if n_converted > 0:
        logger.info(f" - {n_converted} gbk files were converted to fasta files")
    if n_excluded > 0:
        logger.info(
            f" - {n_excluded} gbk files were excluded due to the file name containing '{' or '.join(exclude_name)}'"
        )
    if n_filtered > 0:
        logger.info(f" - {n_filtered} gbk files were excluded due to being at contig edges")
    if n_failed > 0:
        logger.info(f" - {n_failed} gbk files failed to be converted to fasta files")

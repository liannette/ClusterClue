import re

from collections import Counter
from typing import Tuple, Set
from multiprocessing import Pool
from functools import partial

from subclue.preprocess.utils import (
    read_txt,
)

def extract_domain_base_name(domain: str) -> str:
    """Extract the base name of a domain.

    This function extracts the base name of a protein_domain,
    excluding any '_c' suffix of subPfams.

    Args:
        domain (str): The domain name.

    Returns:
        str: The base name of the domain.
    """
    return re.sub(r"_c\d+$", "", domain)


def perform_domain_filtering(
    in_file_path: str,
    domains_to_include_file_path: str,
    out_file_path: str,
    verbose: bool,
) -> str:
    """
    Wrapper for domain filtering of clusters.

    Args:
        in_file_path (str): Path to the input file containing tokenised clusters.
        domains_to_include_file_path (str): Path to the file containing the list of domains to include.
        out_file_path (str): Path to the output file for writing the domain-filtered clusters.
        verbose (bool): If True, print verbose output.
    """
    if verbose:
        print(f"\nPerforming domain filtering on {in_file_path}")
        print(f"Only keeping protein domains listed in {domains_to_include_file_path}")

    include_domains = set(read_txt(domains_to_include_file_path))

    with open(in_file_path, "r") as infile, open(out_file_path, "w") as outfile:
        # header
        header = infile.readline()
        outfile.write(header)
        # for each line/domain
        for line in infile:
            fields = line.rstrip().split("\t")
            domain = fields[6]
            if extract_domain_base_name(domain) in include_domains:
                outfile.write(line)

    if verbose:
        print(f"\nPerformed domain filtering.")
        print(f"The filtered domains have been saved to {out_file_path}")

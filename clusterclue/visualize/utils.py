import sys
import csv
from collections import defaultdict
from pathlib import Path
from typing import List


def read_compounds(compounds_filepath):
    compounds = defaultdict(list)
    with open(compounds_filepath, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            bgc_id, name, smiles = row[0], row[1], row[2]
            compounds[bgc_id].append((name, smiles))
    return compounds


def read_color_domains_file(domains_color_file):
    """
    Reads and parses a color domains file.

    This function reads a tab-delimited file containing domain names and their
    associated RGB color values. Each line in the file should contain a domain
    name followed by a comma-separated RGB value.

    Args:
        domains_color_file (str): Path to the domains color file.

    Returns:
        dict: A dictionary mapping domain accessions (str) to RGB color values (list of 3 integers).

    Raises:
        SystemExit: If the specified file does not exist.

    Example format of domains_color_file:
        Domain1    255,0,0
        Domain2    0,255,0
        Domain3    0,0,255
    """
    if not Path(domains_color_file).is_file():
        sys.exit(f"Error: Domains colors file was not found: {domains_color_file}")

    domain_colors = dict()
    with open(domains_color_file, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            domain_accession = row[0]
            rgb = [int(val) for val in row[1].split(",")]
            domain_colors[domain_accession] = rgb
    return domain_colors


def read_dom_hits(dom_hits_file):
    """
    Reads and parses a domain hits file.

    This function reads a tab-delimited file containing domain hits information.
    Each line in the file should contain information about a domain hit.

    Args:
        dom_hits_file (str): Path to the domain hits file.

    Returns:
        dict: A dictionary mapping gene identifiers (str) to a list of domain hit dictionaries.
            Each domain hit dictionary contains:
                - 'accession' (str): Domain accession.
                - 'start' (int): Start position of the domain relative to the gene start (in bp) and strand direction.
                - 'width' (int): Width of the domain (in bp).

    Raises:
        SystemExit: If the specified file does not exist.

    Example format of dom_hits_file:
        bgc	        g_id	p_id	    location	orf_num	tot_orf domain	    q_range	bitscore
        BGC0000001	orfP	AEK75490.1	0;1083;+	1	    29	    PCMT	    23;143	89.4
        BGC0000001	abyR	AEK75492.1	1886;2633;+	3	    29	    Trans_reg_C	17;87	36.9
        BGC0000001	abyR	AEK75492.1	1886;2633;+	3	    29	    BTAD	    93;238	126.7
    """

    if not Path(dom_hits_file).is_file():
        sys.exit(f"Error: Domain hits file not found: {dom_hits_file}")

    all_domains = defaultdict(list)
    with open(dom_hits_file, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            # gene location (in bp)
            orf_start, orf_end, orf_strand = (
                row["location"].replace("<", "").replace(">", "").split(";")
            )
            orf_start, orf_end = int(orf_start), int(orf_end)

            # domain location (relative to gene start)
            # multiply by 3 to convert aa to bp
            domain_start, domain_end = [
                3 * int(aa_loc) for aa_loc in row["q_range"].split(";")
            ]
            domain_width = domain_end - domain_start
            # get domain start relative to gene direction (strand)
            if orf_strand == "+":
                # This start is relative to the start of the gene
                start = domain_start
            elif orf_strand == "-":
                start = orf_end - orf_start - domain_start - domain_width
            else:
                sys.exit(f"Error: unknown strand {orf_strand}")

            orf_identifier = f"{row['bgc']}_{row['orf_num']}"
            all_domains[orf_identifier].append(
                {
                    "accession": row["domain"],
                    "start": start,
                    "width": domain_width,
                }
            )

    return all_domains


def read_txt(infile_path: str) -> List[str]:
    """Reads a text file into a list of strings, stripping whitespace.

    Args:
        in_file (str): Path to the input file.

    Returns:
        list of str: A list of lines from the file, with leading and trailing whitespace removed.
    """
    return [line.strip() for line in open(infile_path, "r")]


def read_detected_motifs(filename):
    hits = defaultdict(list)
    with open(filename, "r") as infile:
        reader = csv.DictReader(infile, delimiter="\t")
        for row in reader:
            bgc = row["bgc_id"]
            hit = {
                "motif_id": row["motif_id"],
                "n_matches": int(row["n_training"]),
                "threshold": row["score_threshold"],
                "score": row["score"],
                "genes": row["genes"].split(","),
            }
            hits[bgc].append(hit)
    return hits

import sys
import csv
from collections import defaultdict, OrderedDict
from pathlib import Path
from typing import List
from colorsys import hsv_to_rgb, rgb_to_hsv
from random import uniform

from clusterclue.visualize.config import internal_domain_margin


def new_color(gene_or_domain):
    # see https://en.wikipedia.org/wiki/HSL_and_HSV
    # and http://stackoverflow.com/a/1586291

    h = uniform(0, 1)  # all possible colors

    if gene_or_domain == "gene":
        s = uniform(0.15, 0.3)
        v = uniform(0.98, 1.0)
    elif gene_or_domain == "domain":
        s = uniform(0.5, 0.75)  # lower: less saturated
        v = uniform(0.7, 0.9)  # lower = darker
    else:
        sys.exit("unknown kind of color. Should be 'gene' or 'domain'")

    r, g, b = tuple(int(c * 255) for c in hsv_to_rgb(h, s, v))

    return [r, g, b]


def read_color_domains_file(domains_color_file):
    color_domains = OrderedDict()
    if Path(domains_color_file).is_file():
        with open(domains_color_file, "r") as color_domains_handle:
            for line in color_domains_handle:
                # handle comments and empty lines
                if line[0] != "#" and line.strip():
                    row = line.strip().split("\t")
                    name = row[0]
                    rgb = row[1].split(",")
                    color_domains[name] = [int(rgb[x]) for x in range(3)]
    else:
        print("Domains colors file was not found. A new file will be created")
    return color_domains


def read_dom_hits(dom_hits_file, domains_color_file, scaling=30, H=30):
    """Returns dict of {gene_identifier:[[domain_info]]}"""

    if not Path(dom_hits_file).is_file():
        sys.exit(f"Error: {dom_hits_file} not found")

    domain_colors = read_color_domains_file(domains_color_file)

    all_domains = defaultdict(list)
    with open(dom_hits_file, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for fields in reader:
            # gene location (in bp)
            orf_start, orf_end, orf_strand = (
                fields["location"].replace("<", "").replace(">", "").split(";")
            )
            orf_start, orf_end = int(orf_start), int(orf_end)
            # domain location (relative to gene start)
            # multiply by 3 to convert aa to bp
            domain_start, domain_end = [
                3 * int(f) for f in fields["q_range"].split(";")
            ]
            domain_width = domain_end - domain_start

            # get domain start relative to gene direction (strand)
            if orf_strand == "+":
                # This start is relative to the start of the gene
                start = domain_start
            elif orf_strand == "-":
                start = orf_end - orf_start - domain_start - domain_width  # domain_end?
            else:
                sys.exit(f"Error: unknown strand {orf_strand}")

            # colors
            fill_rgb = domain_colors[fields["domain"]]
            # contour color is a bit darker. We go to h,s,v space for that
            h_, s, v = rgb_to_hsv(
                float(fill_rgb[0]) / 255.0,
                float(fill_rgb[1]) / 255.0,
                float(fill_rgb[2]) / 255.0,
            )
            stroke_rgb = tuple(int(c * 255) for c in hsv_to_rgb(h_, s, 0.8 * v))

            cds_identifier = f"{fields['bgc']}_{fields['orf_num']}"
            all_domains[cds_identifier].append(
                {
                    "start": int(start / scaling),
                    "width": int(domain_width / scaling),
                    "height": int(H - 2 * internal_domain_margin),
                    "accession": fields["domain"],
                    "fill_rgb": fill_rgb,
                    "stroke_rgb": stroke_rgb,
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

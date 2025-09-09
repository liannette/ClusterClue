import os
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

    color_domains = read_color_domains_file(domains_color_file)

    if not Path(dom_hits_file).is_file():
        sys.exit("Error (Arrower): " + dom_hits_file + " not found")

    pfam_info = {}
    new_color_domains = {}
    identifiers = defaultdict(list)
    with open(dom_hits_file, "r") as pfd_handle:
        pfd_handle.readline()  # header
        for line in pfd_handle:
            row = line.strip("\n").split("\t")

            # use to access to parent's properties
            # identifier = row[9].replace("<","").replace(">","")
            # if it's the new version of pfd file, we can take the last part
            #  to make it equal to the identifiers used in gene_list. Strand
            #  is recorded in parent gene anyway
            # if ":strand:+" in identifier:
            # identifier = identifier.replace(":strand:+", "")
            # strand = "+"
            # if ":strand:-" in identifier:
            # identifier = identifier.replace(":strand:-", "")
            # strand = "-"
            # get strand
            loc = row[3].replace("<", "").replace(">", "")
            g_start, g_end, strand = loc.split(";")

            # get start and end of pfam
            pf_start, pf_end = row[-2].split(";")
            width = 3 * (int(pf_end) - int(pf_start))

            if strand == "+":
                # multiply by 3 because the env. coordinate is in aminoacids, not in bp
                # This start is relative to the start of the gene
                start = 3 * int(pf_start)
            else:
                loci_start = int(g_start)
                loci_end = int(g_end)

                start = loci_end - loci_start - 3 * int(pf_start) - width

            # geometry
            start = int(start / scaling)
            width = int(width / scaling)

            # accession -> this is now id
            domain_acc = row[6]

            # colors
            try:
                color = color_domains[domain_acc]
            except KeyError:
                color = new_color("domain")
                new_color_domains[domain_acc] = color
                color_domains[domain_acc] = color
                pass
            # contour color is a bit darker. We go to h,s,v space for that
            h_, s, v = rgb_to_hsv(
                float(color[0]) / 255.0,
                float(color[1]) / 255.0,
                float(color[2]) / 255.0,
            )
            color_contour = tuple(int(c * 255) for c in hsv_to_rgb(h_, s, 0.8 * v))

            # [X, L, H, domain_acc, color, color_contour]
            identifier = row[0] + "_" + row[4]
            desc = pfam_info.get(domain_acc, ("", ""))
            identifiers[identifier].append(
                [
                    start,
                    width,
                    int(H - 2 * internal_domain_margin),
                    domain_acc,
                    desc,
                    color,
                    color_contour,
                ]
            )

    if new_color_domains:
        # Save all colors to new file
        new_domains_color_file = domains_color_file + ".new"
        with open(new_domains_color_file, "w") as f:
            for domain_acc, color in color_domains.items():
                color_string = ",".join(map(str, color))
                f.write(f"{domain_acc}\t{color_string}\n")

    return identifiers


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

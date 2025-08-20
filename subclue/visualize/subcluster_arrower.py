#! /usr/bin/python
"""
######################################################################
#                                                                    #
#           PLOT ARROWS FOR GENE CLUSTER GIVEN A GenBank FILE        #
#                           Peter Cimermancic                        #
#                               April 2010                           #
#                heavily modified by Jorge Navarro 2016              #
#                    modified by Joris Louwen 2019                   #
#               for the purpose of plotting sub-clusters             #
######################################################################

Note:
    -Only handles first locus from given gbk
    -Currently all domain-combinations from a sub-cluster are visualised,
        so if a domain-combination from a sub-cluster occurs multiple times
        in a BGC, all those domain-combinations are visualised in the
        sub-cluster.
    -For each BGC, this script loads a given gbk file again for each
        sub-cluster detected in the BGC. Script will get faster if this is
        resolved.
"""

# Makes sure the script can be used with Python 2 as well as Python 3.
from __future__ import division, print_function

from sys import version_info

if version_info[0] == 2:
    range = xrange  # type: ignore  # noqa: F821

import argparse
import os
import re
import sys
from collections import defaultdict
from math import atan2, pi, sin
from random import uniform
from pathlib import Path

from Bio import SeqIO  # type: ignore

from subclue.visualize.utils import (
    read_txt, 
    read_detected_motifs, 
    read_dom_hits, 
    read_color_domains_file
)

from subclue.visualize.molecule import draw_mibig_compounds

from subclue.visualize.config import (
    internal_domain_margin,
    domain_contour_thickness,
    gene_contour_thickness,
    stripe_thickness,
)


# --- Draw arrow for gene
def draw_arrow(
    additional_tabs,
    X,
    Y,
    L,
    l,  # noqa: E741, E741
    H,
    h,
    strand,
    color,
    color_contour,
    category,
    gid,
    domain_list,
):
    """
    SVG code for arrow:
        - (X,Y) ... upper left (+) or right (-) corner of the arrow
        - L ... arrow length
        - H ... arrow height
        - strand
        - h ... arrow head edge width
        - l ... arrow head length
        - color
        - strand
    the edges are ABCDEFG starting from (X,Y)
    domain_list: list of elements to draw domains
    """

    if strand == "+":
        head_end = L
        if L < l:
            # squeeze arrow if length shorter than head length
            A = [X, Y - h]
            B = [X + L, Y + H / 2]
            C = [X, Y + H + h]
            head_start = 0
            points = [A, B, C]
        else:
            A = [X, Y]
            B = [X + L - l, Y]
            C = [X + L - l, Y - h]
            D = [X + L, Y + H / 2]
            E = [X + L - l, Y + H + h]
            F = [X + L - l, Y + H]
            G = [X, Y + H]
            head_start = (
                L - l
            )  # relative to the start of the gene, not absolute coords.
            points = [A, B, C, D, E, F, G]

    elif strand == "-":
        head_start = 0
        if L < l:
            # squeeze arrow if length shorter than head length
            A = [X, Y + H / 2]
            B = [X + L, Y - h]
            C = [X + L, Y + H + h]
            head_end = L
            points = [A, B, C]
        else:
            A = [X + L, Y]
            B = [X + l, Y]
            C = [X + l, Y - h]
            D = [X, Y + H / 2]
            E = [X + l, Y + H + h]
            F = [X + l, Y + H]
            G = [X + L, Y + H]
            head_end = l
            points = [A, B, C, D, E, F, G]

    else:
        return ""

    head_length = head_end - head_start
    if head_length == 0:
        return ""

    points_coords = []
    for point in points:
        points_coords.append(str(int(point[0])) + "," + str(int(point[1])))

    arrow = additional_tabs + "\t<g>\n"

    # unidentified genes don't have a title and have a darker contour
    if gid != "NoName":
        arrow += additional_tabs + "\t\t<title>" + gid + "</title>\n"
    else:
        color_contour = [50, 50, 50]

    arrow += '{}\t\t<polygon class="{}" '.format(additional_tabs, gid)
    arrow += 'points="{}" fill="rgb({})" '.format(
        " ".join(points_coords), ",".join([str(val) for val in color])
    )
    arrow += 'fill-opacity="1.0" stroke="rgb({})" '.format(
        ",".join([str(val) for val in color_contour])
    )
    arrow += 'stroke-width="{}" {} />\n'.format(str(gene_contour_thickness), category)

    # paint domains. Domains on the tip of the arrow should not have corners sticking
    #  out of them
    for domain in domain_list:
        # [X, L, H, domain_accession, (domain_name, domain_description), color, color_contour]
        dX = domain[0]
        dL = domain[1]
        dH = domain[2]
        dacc = domain[3]
        dname = domain[4][0]
        ddesc = domain[4][1]
        dcolor = domain[5]
        dccolor = domain[6]

        arrow += additional_tabs + "\t\t<g>\n"
        arrow += '{}\t\t\t<title>{} ({})\n"{}"</title>\n'.format(
            additional_tabs, dname, dacc, ddesc
        )

        if strand == "+":
            # calculate how far from head_start we (the horizontal guide at y=Y+internal_domain_margin)
            #  would crash with the slope
            # Using similar triangles:
            collision_x = head_length * (h + internal_domain_margin)
            collision_x /= h + H / 2.0
            collision_x = round(collision_x)

            # either option for x_margin_offset work
            # m = -float(h + H/2)/(head_length) #slope of right line
            # x_margin_offset = (internal_domain_margin*sqrt(1+m*m))/m
            # x_margin_offset = -(x_margin_offset)
            x_margin_offset = internal_domain_margin / sin(
                pi - atan2(h + H / 2.0, -head_length)
            )

            if (dX + dL) < head_start + collision_x - x_margin_offset:
                arrow += '{}\t\t\t<rect class="{}" x="{}" '.format(
                    additional_tabs, dacc, str(X + dX)
                )
                arrow += 'y="{}" stroke-linejoin="round" '.format(
                    str(Y + internal_domain_margin)
                )
                arrow += 'width="{}" height="{}" '.format(str(dL), str(dH))
                arrow += 'fill="rgb({})" stroke="rgb({})" '.format(
                    ",".join([str(val) for val in dcolor]),
                    ",".join([str(val) for val in dccolor]),
                )
                arrow += 'stroke-width="{}" opacity="0.75" />\n'.format(
                    str(domain_contour_thickness)
                )
            else:
                del points[:]

                if dX < head_start + collision_x - x_margin_offset:
                    # add points A and B
                    points.append([X + dX, Y + internal_domain_margin])
                    points.append(
                        [
                            X + head_start + collision_x - x_margin_offset,
                            Y + internal_domain_margin,
                        ]
                    )

                else:
                    # add point A'
                    start_y_offset = (h + H / 2) * (L - x_margin_offset - dX)
                    start_y_offset /= head_length
                    start_y_offset = int(start_y_offset)
                    points.append([X + dX, int(Y + H / 2 - start_y_offset)])

                # handle the rightmost part of the domain
                if (
                    dX + dL >= head_end - x_margin_offset
                ):  # could happen more easily with the scaling
                    points.append(
                        [X + head_end - x_margin_offset, int(Y + H / 2)]
                    )  # right part is a triangle
                else:
                    # add points C and D
                    end_y_offset = (2 * h + H) * (L - x_margin_offset - dX - dL)
                    end_y_offset /= 2 * head_length
                    end_y_offset = int(end_y_offset)

                    points.append([X + dX + dL, int(Y + H / 2 - end_y_offset)])
                    points.append([X + dX + dL, int(Y + H / 2 + end_y_offset)])

                # handle lower part
                if dX < head_start + collision_x - x_margin_offset:
                    # add points E and F
                    points.append(
                        [
                            X + head_start + collision_x - x_margin_offset,
                            Y + H - internal_domain_margin,
                        ]
                    )
                    points.append([X + dX, Y + H - internal_domain_margin])
                else:
                    # add point F'
                    points.append([X + dX, int(Y + H / 2 + start_y_offset)])

                del points_coords[:]
                for point in points:
                    points_coords.append(str(int(point[0])) + "," + str(int(point[1])))

                arrow += '{}\t\t\t<polygon class="{}" '.format(additional_tabs, dacc)
                arrow += 'points="{}" stroke-linejoin="round" '.format(
                    " ".join(points_coords)
                )
                arrow += 'width="{}" height="{}" '.format(str(dL), str(dH))
                arrow += 'fill="rgb({})" '.format(
                    ",".join([str(val) for val in dcolor])
                )
                arrow += 'stroke="rgb({})" '.format(
                    ",".join([str(val) for val in dccolor])
                )
                arrow += 'stroke-width="{}" opacity="0.75" />\n'.format(
                    str(domain_contour_thickness)
                )

        # now check other direction
        else:
            # calculate how far from head_start we (the horizontal guide at y=Y+internal_domain_margin)
            #  would crash with the slope
            # Using similar triangles:
            collision_x = head_length * ((H / 2) - internal_domain_margin)
            collision_x /= h + H / 2.0
            collision_x = round(collision_x)

            x_margin_offset = round(
                internal_domain_margin / sin(atan2(h + H / 2.0, head_length))
            )

            # nice, blocky domains
            if dX > collision_x + x_margin_offset:
                arrow += '{}\t\t\t<rect class="{}" '.format(additional_tabs, dacc)
                arrow += 'x="{}" y="{}" '.format(
                    str(X + dX), str(Y + internal_domain_margin)
                )
                arrow += 'stroke-linejoin="round" width="{}" height="{}" '.format(
                    str(dL), str(dH)
                )
                arrow += 'fill="rgb({})" '.format(
                    ",".join([str(val) for val in dcolor])
                )
                arrow += 'stroke="rgb({})" '.format(
                    ",".join([str(val) for val in dccolor])
                )
                arrow += 'stroke-width="{}" opacity="0.75" />\n'.format(
                    str(domain_contour_thickness)
                )
            else:
                del points[:]

                # handle lefthand side. Regular point or pointy start?
                if dX >= x_margin_offset:
                    start_y_offset = round(
                        (h + H / 2) * (dX - x_margin_offset) / head_length
                    )
                    points.append([X + dX, Y + H / 2 - start_y_offset])
                else:
                    points.append([X + x_margin_offset, Y + H / 2])

                # handle middle/end
                if dX + dL < collision_x + x_margin_offset:
                    if head_length != 0:
                        end_y_offset = round(
                            (h + H / 2) * (dX + dL - x_margin_offset) / head_length
                        )
                    else:
                        end_y_offset = 0
                    points.append([X + dX + dL, Y + H / 2 - end_y_offset])
                    points.append([X + dX + dL, Y + H / 2 + end_y_offset])
                else:
                    points.append(
                        [X + collision_x + x_margin_offset, Y + internal_domain_margin]
                    )
                    points.append([X + dX + dL, Y + internal_domain_margin])
                    points.append([X + dX + dL, Y + internal_domain_margin + dH])
                    points.append(
                        [
                            X + collision_x + x_margin_offset,
                            Y + internal_domain_margin + dH,
                        ]
                    )

                # last point, if it's not a pointy domain
                if dX >= x_margin_offset:
                    points.append([X + dX, Y + H / 2 + start_y_offset])

                del points_coords[:]
                for point in points:
                    points_coords.append(str(int(point[0])) + "," + str(int(point[1])))

                arrow += '{}\t\t\t<polygon class="{}" '.format(additional_tabs, dacc)
                arrow += 'points="{}" stroke-linejoin="round" '.format(
                    " ".join(points_coords)
                )
                arrow += 'width="{}" height="{}" '.format(str(dL), str(dH))
                arrow += 'fill="rgb({})" '.format(
                    ",".join([str(val) for val in dcolor])
                )
                arrow += 'stroke="rgb({})" '.format(
                    ",".join([str(val) for val in dccolor])
                )
                arrow += 'stroke-width="{}" opacity="0.75" />\n'.format(
                    str(domain_contour_thickness)
                )

        arrow += additional_tabs + "\t\t</g>\n"

    arrow += additional_tabs + "\t</g>\n"

    return arrow


def draw_line(X, Y, L):
    """
    Draw a line below genes
    """

    line = '<line x1="{}" y1="{}" x2="{}" y2="{}" style="stroke:rgb(70,70,70); stroke-width:{} "/>\n'.format(
        str(X), str(Y), str(X + L), str(Y), str(stripe_thickness)
    )

    return line


def draw_bgc(
    bgc_gbk_path,
    all_domains,
    gene_colors=None,
    motif_hit=None,
    included_domains=None,
    H=30,
    h=5,
    l=12,  # noqa: E741
    mX=10,
    mY=10,
    scaling=30,
    html=True,
):
    """
    Draw the BGC or the detected motif in SVG format.
    """
    bgc_id = bgc_gbk_path.stem
    seq_record = list(SeqIO.parse(bgc_gbk_path, "genbank"))[0]

    # -- Create SVG header
    if motif_hit:
        text = (
            f"Motif: {motif_hit['motif_id']} "
            f"(n: {motif_hit['n_matches']}, threshold: {motif_hit['threshold']}), "
            f"score: {motif_hit['score']}, n_genes: {len(motif_hit['genes'])}"
        )
        # Smaller font for motif ID (h2)
        header = (
            f'<div><h2>{text}</h2></div>\n'
            f'\t\t<div title="{motif_hit['motif_id']}">\n'
        )
    else:
        # Bigger font for BGC ID (h1)
        header = (
            f'<div><h1>{bgc_id}</h1></div>\n'
            f'\t\t<div title="{bgc_id}">\n'
        )
    svg_width = len(seq_record)/scaling + 2 * mX
    svg_height = 2 * h + H + 2 * mY
    if html:
        header += f'\t\t\t<svg width="{svg_width}" height="{svg_height}">\n'
        add_tabs = "\t\t\t"
    else:
        # SVG header for non-HTML output has no title
        header = (
            f'<svg version="1.1" baseProfile="full" xmlns="http://www.w3.org/2000/svg" '
            f'width="{svg_width}" height="{svg_height}">\n'
        )
        add_tabs = "\t"
    svg_text = header


    # --- draw the BGC
    svg_text += f'{add_tabs}<g>\n'

    # draw a line that corresponds to cluster size
    line = draw_line(mX, mY + h + H/2, len(seq_record)/scaling)
    svg_text += f'{add_tabs}\t{line}'

    # Draw arrows for each CDS feature
    color_contour = (0, 0, 0)
    cds_num = 0

    if motif_hit:
        hit_protein_ids = dict()
        for tokenised_gene, p_ids in motif_hit["genes"].items():
            for p_id in p_ids:
                hit_protein_ids[p_id] = tokenised_gene.split(";")

    for feature in seq_record.features:
        # Check if the feature is CDS
        if feature.type != "CDS":
            continue
        cds_num += 1

        if motif_hit:
            # Skip cds if not part of the detected motif
            protein_id = feature.qualifiers.get("protein_id", [""])[0]
            if protein_id not in hit_protein_ids:
                continue
            # Dont color the gene, but the domains in the hit
            domains_in_gene = all_domains[f"{bgc_id}_{cds_num}"] # [X, L, H, domain_acc, color, color_contour]
            # Get all domains of the hit
            domain_list = [d for d in domains_in_gene if d[3] in hit_protein_ids[protein_id]]
            color_fill = (255, 255, 255)
        else:
            # Change this to color according to gene function/kind
            gene_kind = feature.qualifiers.get("gene_kind", ["other"])[0]
            color_fill = gene_colors.get(gene_kind, (255, 255, 255))
            domain_list = []


        # Create a tag for CDS
        cds_tag = ""
        if "gene" in feature.qualifiers:
            cds_tag += feature.qualifiers["gene"][0]
        if "locus_tag" in feature.qualifiers:
            locus_tag = feature.qualifiers["locus_tag"][0]
            cds_tag += f" ({locus_tag})"
        if "product" in feature.qualifiers:
            product = feature.qualifiers["product"][0]
            cds_tag += f"\n{product}"

        # Convert numerical strand to string representation
        strand = feature.location.strand
        strand = "+" if strand == 1 else "-" if strand == -1 else sys.exit(f"Invalid strand: {strand}")

        # define arrow's start and end
        # http://biopython.org/DIST/docs/api/Bio.SeqFeature.FeatureLocation-class.html#start
        arrow_start = int(feature.location.start) / scaling
        arrow_end = int(feature.location.end) / scaling
        arrow_length = arrow_end - arrow_start

        arrow = draw_arrow(
            additional_tabs=add_tabs,
            X=arrow_start + mX,
            Y=mY + h,
            L=arrow_length,
            l=l,
            H=H,
            h=h,
            strand=strand,
            color=color_fill,
            color_contour=color_contour,
            category="",
            gid=cds_tag,
            domain_list=domain_list,
        )
        if arrow == "":
            print(f"  (ArrowerSVG) Warning: something went wrong with {bgc_id}")

        svg_text += arrow

    svg_text += f"{add_tabs}</g>\n"

    # Close the SVG tag
    svg_text += f"{add_tabs[:-2]}</svg>\n"
    if html:
        svg_text += "\t\t</div>\n"

    return svg_text


def main(
    filenames,
    dom_hits_file,
    one=False,
    include_list=None,
    domains_color_file=None,
    outfile="output.html",
    motif_hits=None,
    json_dir="",
    verbose=False,
):
    # # depreciated variables
    # only_color_genes = False
    # gene_colors = {}

    # Read BGC paths
    if one: 
        bgc_gbk_paths = [Path(filenames), ] 
    else:
        bgc_gbk_paths = [Path(path) for path in read_txt(filenames)]

    all_domains = read_dom_hits(dom_hits_file, domains_color_file)
    include_doms = read_txt(include_list) if include_list else None
    detected_motifs = read_detected_motifs(motif_hits) if motif_hits else None

    # standard antismash colors
    gene_colors = {
        "biosynthetic": (129, 14, 21),
        "biosynthetic-additional": (241, 109, 117),
        "transport": (241, 109, 117),
        "regulatory": (46, 139, 87),
        "other": (128, 128, 128),
    }

    if verbose:
        print("\nVisualising sub-clusters")

    with open(outfile, "w") as f:

        bgc_gbk_paths = bgc_gbk_paths[:5]
        for bgc_path in bgc_gbk_paths:
            # Draw the molecule structure if MIBiG BGC
            json_file = Path(json_dir) / f"{bgc_path.stem}.json"
            if json_file.is_file():
                svg_text = draw_mibig_compounds(json_file)
                f.write(svg_text)

            # Draw the BGC
            svg_text = draw_bgc(
                bgc_gbk_path=bgc_path,
                all_domains=all_domains,
                gene_colors=gene_colors
            )
            f.write(svg_text)

            # Draw the detected motifs if available
            for motif_hit in detected_motifs.get(bgc_path.stem, []):
                svg_text = draw_bgc(
                    bgc_gbk_path=bgc_path,
                    all_domains=all_domains,
                    motif_hit=motif_hit,
                    included_domains=include_doms
                )
                f.write(svg_text)

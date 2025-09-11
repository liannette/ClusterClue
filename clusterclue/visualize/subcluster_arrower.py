"""
######################################################################
#                                                                    #
#           PLOT ARROWS FOR GENE CLUSTER GIVEN A GenBank FILE        #
#                           Peter Cimermancic                        #
#                               April 2010                           #
#                heavily modified by Jorge Navarro 2016              #
#                    modified by Joris Louwen 2019                   #
#             again heavily modified by Annette Lien 2025            #
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

import re
import sys
from math import atan2, pi, sin
from pathlib import Path
from Bio import SeqIO  # type: ignore

from clusterclue.visualize.molecule import read_compounds, draw_compounds
from clusterclue.visualize.config import (
    internal_domain_margin,
    domain_contour_thickness,
    gene_contour_thickness,
    stripe_thickness,
)
from clusterclue.visualize.utils import (
    read_txt,
    read_detected_motifs,
    read_dom_hits,
)


def _get_gene_coordinates(X, Y, L, l, H, h, strand):
    if strand == "+":
        if L < l:
            # squeeze arrow if length shorter than head length
            A = [X, Y - h]
            B = [X + L, Y + H / 2]
            C = [X, Y + H + h]
            points = [A, B, C]
        else:
            A = [X, Y]
            B = [X + L - l, Y]
            C = [X + L - l, Y - h]
            D = [X + L, Y + H / 2]
            E = [X + L - l, Y + H + h]
            F = [X + L - l, Y + H]
            G = [X, Y + H]
            points = [A, B, C, D, E, F, G]

    elif strand == "-":
        if L < l:
            # squeeze arrow if length shorter than head length
            A = [X, Y + H / 2]
            B = [X + L, Y - h]
            C = [X + L, Y + H + h]
            points = [A, B, C]
        else:
            A = [X + L, Y]
            B = [X + l, Y]
            C = [X + l, Y - h]
            D = [X, Y + H / 2]
            E = [X + l, Y + H + h]
            F = [X + l, Y + H]
            G = [X + L, Y + H]
            points = [A, B, C, D, E, F, G]
    return points


def _get_arrow_head_location(L, l, strand):
    if strand == "+":
        head_end = L
        # no tail
        if L < l:
            head_start = 0
        # tail
        else:
            head_start = L - l  # relative to start of gene, not absolute coords.
    elif strand == "-":
        head_start = 0
        # no tail
        if L < l:
            head_end = L
        # tail
        else:
            head_end = l

    return head_start, head_end


def _get_domain_coordinates(
    dX, dL, dH, X, Y, L, H, h, strand, head_start, head_end, head_length
):
    points = []
    if strand == "+":
        # calculate how far from head_start we would crash with the slope
        # (the horizontal guide at y=Y+internal_domain_margin)
        # Using similar triangles:
        collision_x = head_length * (h + internal_domain_margin)
        collision_x /= h + H / 2.0
        collision_x = round(collision_x)

        # either option for x_margin_offset work
        # m = -float(h + H/2)/(head_length) #slope of right line
        # x_margin_offset = (internal_domain_margin*sqrt(1+m*m))/m
        # x_margin_offset = -(x_margin_offset)
        x_margin_offset = round(
            internal_domain_margin / sin(pi - atan2(h + H / 2.0, -head_length))
        )

        # no collision -> nice, blocky domains
        if (dX + dL) < head_start + collision_x - x_margin_offset:
            points.extend(
                [
                    [X + dX, Y + internal_domain_margin],
                    [X + dX + dL, Y + internal_domain_margin],
                    [X + dX + dL, Y + internal_domain_margin + dH],
                    [X + dX, Y + internal_domain_margin + dH],
                ]
            )
        # collision -> draw a polygon
        else:
            points = []

            # handle the left part of domain (tail)
            if dX < head_start + collision_x - x_margin_offset:
                # arrow with tail: add points A and B
                points.append([X + dX, Y + internal_domain_margin])
                points.append(
                    [
                        X + head_start + collision_x - x_margin_offset,
                        Y + internal_domain_margin,
                    ]
                )
            else:
                # arrow without tail: add point A'
                start_y_offset = int(
                    (h + H / 2) * (L - x_margin_offset - dX) / head_length
                )
                points.append([X + dX, int(Y + H / 2 - start_y_offset)])

            # handle the right part of domain (arrow head)
            if dX + dL >= head_end - x_margin_offset:  # could happen with scaling
                # right part is a triangle
                points.append([X + head_end - x_margin_offset, int(Y + H / 2)])
            else:
                # right part is a cut triangle
                end_y_offset = (2 * h + H) * (L - x_margin_offset - dX - dL)
                end_y_offset /= 2 * head_length
                end_y_offset = int(end_y_offset)
                points.append([X + dX + dL, int(Y + H / 2 - end_y_offset)])
                points.append([X + dX + dL, int(Y + H / 2 + end_y_offset)])

            # handle lower part
            if dX < head_start + collision_x - x_margin_offset:
                # arrow with tail: add points E and F
                points.append(
                    [
                        X + head_start + collision_x - x_margin_offset,
                        Y + H - internal_domain_margin,
                    ]
                )
                points.append([X + dX, Y + H - internal_domain_margin])
            else:
                # # arrow without tail: add point F'
                points.append([X + dX, int(Y + H / 2 + start_y_offset)])

    # now check other direction (strand == "-")
    elif strand == "-":
        # calculate how far from head_start we would crash with the slope
        # (the horizontal guide at y=Y+internal_domain_margin)
        # Using similar triangles:
        collision_x = head_length * ((H / 2) - internal_domain_margin)
        collision_x /= h + H / 2.0
        collision_x = round(collision_x)

        x_margin_offset = round(
            internal_domain_margin / sin(atan2(h + H / 2.0, head_length))
        )

        # no collision -> nice, blocky domains
        if dX > collision_x + x_margin_offset:
            points.extend(
                [
                    [X + dX, Y + internal_domain_margin],
                    [X + dX + dL, Y + internal_domain_margin],
                    [X + dX + dL, Y + internal_domain_margin + dH],
                    [X + dX, Y + internal_domain_margin + dH],
                ]
            )
        # collision -> draw a polygon
        else:
            # handle left part of domain (head)
            if dX < x_margin_offset:
                # regular triangle
                points.append([X + x_margin_offset, Y + H / 2])
            else:
                # cut triangle
                start_y_offset = round(
                    (h + H / 2) * (dX - x_margin_offset) / head_length
                )
                points.append([X + dX, Y + H / 2 + start_y_offset])
                points.append([X + dX, Y + H / 2 - start_y_offset])

            # handle middle/end
            if dX + dL < collision_x + x_margin_offset:
                # no tail
                if head_length != 0:
                    end_y_offset = round(
                        (h + H / 2) * (dX + dL - x_margin_offset) / head_length
                    )
                else:
                    end_y_offset = 0
                points.append([X + dX + dL, Y + H / 2 - end_y_offset])
                points.append([X + dX + dL, Y + H / 2 + end_y_offset])
            else:
                # tail
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
    return points


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
    only_color_genes: bool, only color genes
    """
    if strand not in ["+", "-"]:
        return ""

    arrow_head_start, arrow_head_end = _get_arrow_head_location(L, l, strand)
    arrow_head_length = arrow_head_end - arrow_head_start

    if arrow_head_length == 0:
        return ""

    svg_str = f"{additional_tabs}<g>\n"

    gene_points = _get_gene_coordinates(X, Y, L, l, H, h, strand)
    gid = gid.replace("\n", " | ")

    svg_str += f"{additional_tabs}\t<title>{gid}</title>\n"
    svg_str += (
        f'{additional_tabs}\t<polygon class="{gid}" '
        f'points="{" ".join(f"{point[0]},{point[1]}" for point in gene_points)}" '
        f'fill="rgb({",".join([str(val) for val in color])})" fill-opacity="1.0" '
        f'stroke="rgb({",".join([str(val) for val in color_contour])})" '
        f'stroke-width="{gene_contour_thickness}" />\n'
    )

    for domain in domain_list:
        # [X, L, H, domain_accession, (domain_name, domain_description), color, color_contour]
        domain_X = domain[0]
        domain_L = domain[1]
        domain_H = domain[2]
        domain_acc = domain[3]
        domain_name = domain[4][0]
        domain_desc = domain[4][1]
        domain_fill = domain[5]
        domain_stroke = domain[6]

        # Domains on the tip of the arrow should not have corners sticking out
        domain_points = _get_domain_coordinates(
            domain_X,
            domain_L,
            domain_H,
            X,
            Y,
            L,
            H,
            h,
            strand,
            arrow_head_start,
            arrow_head_end,
            arrow_head_length,
        )

        domain_points_str = " ".join(
            f"{point[0]},{point[1]}" for point in domain_points
        )

        svg_str += f"{additional_tabs}\t<g>\n"
        svg_str += f'{additional_tabs}\t\t<title>{domain_name} ({domain_acc}) "{domain_desc}"</title>\n'
        svg_str += (
            f'{additional_tabs}\t\t<polygon class="{domain_acc}" '
            f'points="{domain_points_str}" stroke-linejoin="round" '
            f'width="{domain_L}" height="{domain_H}" '
            f'fill="rgb({",".join([str(val) for val in domain_fill])})" '
            f'stroke="rgb({",".join([str(val) for val in domain_stroke])})" '
            f'stroke-width="{domain_contour_thickness}" opacity="0.75" />\n'
        )
        svg_str += f"{additional_tabs}\t</g>\n"

    # end of gene arrow
    svg_str += f"{additional_tabs}</g>\n"

    return svg_str


def draw_line(X, Y, L):
    """
    Draw a line below genes
    """
    line = '<line x1="{}" y1="{}" x2="{}" y2="{}" style="stroke:rgb(70,70,70); stroke-width:{} "/>\n'.format(
        str(X), str(Y), str(X + L), str(Y), str(stripe_thickness)
    )
    return line


def _get_tokenized_gene(domain_ids, included_domains):
    """
    Remove domains that were not included in the analysis.
    """
    filtered_domains = []
    for domain in domain_ids:
        # get the domain name without subPfam suffix
        # e.g. "PF00001_c1" -> "PF00001"
        # e.g. "PF00001" -> "PF00001"
        match = re.search(r"_c\d+$", domain)
        if match:
            domain_clean = domain[: match.start()]
        else:
            domain_clean = domain
        # check if the domain is in the included domains
        if domain_clean in included_domains:
            filtered_domains.append(domain)
    return ";".join(filtered_domains)


def draw_bgc(
    bgc_gbk_path,
    domain_hits,
    motif_hit=None,
    included_domains=None,
    H=30,
    h=5,
    l=12,  # noqa: E741
    mX=10,
    mY=10,
    scaling=30,
):
    """
    Draw the BGC or the detected motif in SVG format.
    """
    # -- Create SVG header
    header = "<div></div>\n"

    if motif_hit:
        # Smaller font for motif ID
        text = (
            f"Motif: {motif_hit['motif_id']} "
            f"(n: {motif_hit['n_matches']}, threshold: {motif_hit['threshold']}), "
            f"score: {motif_hit['score']}, n_genes: {len(motif_hit['genes'])}"
        )
        header += f"<div><h3>{text}</h3></div>\n"

    svg_text = header

    bgc_id = bgc_gbk_path.stem
    seq_record = list(SeqIO.parse(bgc_gbk_path, "genbank"))[0]

    svg_width = len(seq_record) / scaling + 2 * mX
    svg_height = 2 * h + H + 2 * mY
    svg_text += f'<svg width="{svg_width}" height="{svg_height}">\n'

    add_tabs = "\t"

    # draw a line that corresponds to cluster size
    line = draw_line(mX, mY + h + H / 2, len(seq_record) / scaling)
    svg_text += f"{add_tabs}{line}"

    # Draw arrows for each CDS feature
    color_contour = (0, 0, 0)
    color_fill = (255, 255, 255)
    cds_num = 0
    for feature in seq_record.features:
        # Check if the feature is CDS
        if feature.type != "CDS":
            continue

        cds_num += 1

        # Get the identifier for the domain hits
        identifier = f"{bgc_id}_{cds_num}"
        domain_list = domain_hits[identifier]

        if motif_hit:
            # Skip cds if not part of the detected motif
            cds_domains = [info[3] for info in domain_list]
            tokenized_gene = _get_tokenized_gene(cds_domains, included_domains)
            if tokenized_gene not in motif_hit["genes"]:
                continue

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
        strand = (
            "+"
            if strand == 1
            else "-"
            if strand == -1
            else sys.exit(f"Invalid strand: {strand}")
        )

        # define arrow's start and end
        # http://biopython.org/DIST/docs/api/Bio.SeqFeature.FeatureLocation-class.html#start
        gene_start = int(feature.location.start) / scaling
        gene_end = int(feature.location.end) / scaling
        gene_length = gene_end - gene_start

        arrow = draw_arrow(
            additional_tabs=add_tabs,
            X=gene_start + mX,
            Y=mY + h,
            L=gene_length,
            l=l,
            H=H,
            h=h,
            strand=strand,
            color=color_fill,
            color_contour=color_contour,
            gid=cds_tag,
            domain_list=domain_list,
        )
        if arrow == "":
            print(f"  (ArrowerSVG) Warning: something went wrong with {bgc_id}")

        svg_text += arrow

    # Close the SVG tag
    svg_text += "</svg>\n"

    return svg_text


def main(
    filenames,
    dom_hits_file,
    include_list=None,
    domains_color_file=None,
    outfile="output.html",
    motif_hits=None,
    compounds_filepath=None,
    verbose=False,
):
    # depreciated variables
    only_color_genes = False
    gene_colors = {}

    # Read BGC paths
    bgc_gbk_paths = [Path(path) for path in read_txt(filenames)]

    dom_hits = read_dom_hits(dom_hits_file, domains_color_file)
    include_doms = read_txt(include_list) if include_list else None
    detected_motifs = read_detected_motifs(motif_hits) if motif_hits else None
    compounds = read_compounds(compounds_filepath) if compounds_filepath else None

    if verbose:
        print("\nVisualising sub-clusters")

    with open(outfile, "w") as f:
        for bgc_path in bgc_gbk_paths:
            bgc_id = bgc_path.stem

            # Write header for each BGC
            f.write(f"<h1>{bgc_id}</h1>\n")

            # Draw the molecule structure if available
            if compounds:
                svg_text = draw_compounds(compounds.get(bgc_id, []))
                f.write(svg_text)

            # Draw the full BGC
            svg_text = draw_bgc(
                bgc_gbk_path=bgc_path,
                domain_hits=dom_hits,
            )
            f.write(svg_text)

            # Draw the detected motifs if available
            for motif_hit in detected_motifs.get(bgc_path.stem, []):
                svg_text = draw_bgc(
                    bgc_gbk_path=bgc_path,
                    domain_hits=dom_hits,
                    motif_hit=motif_hit,
                    included_domains=include_doms,
                )
                f.write(svg_text)

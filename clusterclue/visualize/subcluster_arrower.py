#! /usr/bin/python
'''
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
'''
# Makes sure the script can be used with Python 2 as well as Python 3.
from __future__ import print_function, division
from sys import version_info
if version_info[0] == 2:
    range = xrange

import os
import sys
import argparse
from Bio import SeqIO
from random import uniform
from colorsys import hsv_to_rgb
from colorsys import rgb_to_hsv
from math import sin, atan2, pi
from collections import defaultdict
from itertools import groupby
import re

global internal_domain_margin
global gene_contour_thickness
global stripe_thickness
global gene_categories_color

internal_domain_margin = 3
domain_contour_thickness = 1
gene_contour_thickness = 2 # thickness grows outwards
stripe_thickness = 3

# domains_color_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "domains_color_file.tsv")

def get_commands():
    parser = argparse.ArgumentParser(description="A script visualise BGCs and\
        their modules found with the statistical method and with LDA.")
    parser.add_argument("-f","--filenames",help="A file that contains paths to\
        gbk files of BGCs that will be plot, or one gbk file when --one is\
        provided", required=True)
    parser.add_argument("-c", "--domains_colour_file",help='A tsv file that\
        contains domain_id\tr,g,b on each line. Must be specified, but can be\
        an empty file in which domain colours will be added')
    parser.add_argument('-d','--dom_hits_file',help='A file in which Pfam\
        domains are linked to genes: bgc\tg_id\tp_id\tlocation\torf_num\t\
        tot_orf\tdomain\tq_range\tbitscore, location as start;end;strand \
        qrange as start;end')
    parser.add_argument('-g','--genes_colour_file',help='A tsv file that\
        contains BGCname_GeneNumber\tr,g,b on each line. Is optional, but when\
        used domains will not be coloured',default=False)
    parser.add_argument('-o','--outfile',help='Outfile filepath')
    parser.add_argument('-l','--modules_lda',help='A file with matches to\
        topics originating from the LDA algorithm, optional.',default=False)
    parser.add_argument('-t','--topic_include', help='If specified, only this\
        one or more topics will be included in the visualisation',\
        default=False, nargs="+")
    parser.add_argument('-s','--modules_stat',help='A file with BGCs linked\
        to statistical modules as >BGC\nmod_info, optional',default=False)
    parser.add_argument('-i','--include_stat_module', help='If specified, only\
        this one or more stat_module will be included in the visualisation',\
        default=False, nargs="+")
    parser.add_argument('--one', help='Instead of a file containing locations\
        of gbk files, there is one gbk supplied with -f gbkfile.gbk',\
        default=False, action='store_true')
    parser.add_argument('--include_stat_family',help='If specified, only\
        this one or more families will be included in the visualisation',\
        default=False, nargs="+")
    parser.add_argument('--include_stat_clan',help='If specified, only\
        this one or more families will be included in the visualisation',\
        default=False, nargs="+")
    parser.add_argument("--include_list", dest="include_list", default=False, \
        help="If provided only the domains in this file will be taken into \
        account in the plotting of subclusters. One line should contain one \
        Pfam ID (default: False - meaning all Pfams present in domhits file)")
    parser.add_argument(
        "-v", "--verbose", dest="verbose", required=False, action="store_true",
        default=False, help="Prints more detailed information.")
    return parser.parse_args()

# read various color data
def read_color_genes_file(gene_color_file):
    # Try to read already-generated colors for genes
    color_genes = {}
    
    if os.path.isfile(gene_color_file):
        print("  Found file with gene colors")
        with open(gene_color_file, "r") as color_genes_handle:
            for line in color_genes_handle:
                # handle comments and empty lines
                if line[0] != "#" and line.strip():
                    row = line.strip().split("\t")
                    name = row[0]
                    rgb = row[1].split(",")
                    color_genes[name] = [int(rgb[x]) for x in range(3)]
    else:
        print("  Gene color file was not found. A new file will be created")
        with open('gene_color_file.txt', "w") as color_genes_handle:
            color_genes_handle.write("NoName\t255,255,255\n")
        color_genes = {"NoName":[255, 255, 255]}
    
    return color_genes


def read_color_domains_file(domains_color_file):
    # Try to read colors for domains
    color_domains = {}
    
    if os.path.isfile(domains_color_file):
        print("\nFound file with domains colors")
        with open(domains_color_file, "r") as color_domains_handle:
            for line in color_domains_handle:
                # handle comments and empty lines
                if line[0] != "#" and line.strip():
                    row = line.strip().split("\t")
                    name = row[0]
                    rgb = row[1].split(",")
                    color_domains[name] = [int(rgb[x]) for x in range(3)]
    else:
        print("  Domains colors file was not found. An empty file will be created")
        color_domains_handle = open(domains_color_file, "a+")
        
    return color_domains


# Try to read categories:
def read_pfam_domain_categories():
    pfam_category = {}
    
    if os.path.isfile(pfam_domain_categories):
        print("  Found file with Pfam domain categories")
        with open(pfam_domain_categories, "r") as cat_handle:            
            for line in cat_handle:
                # handle comments and empty lines
                if line[0] != "#" and line.strip():
                    row = line.strip().split("\t")
                    domain = row[1]
                    category = row[0]
                    pfam_category[domain] = category
    else:
        print("  File pfam_domain_categories was NOT found")
                    
    return pfam_category
   

# --- Draw arrow for gene
def draw_arrow(additional_tabs, X, Y, L, l, H, h, strand, color,\
    color_contour, category, gid, domain_list, only_color_genes):
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

    if strand == '+':
        head_end = L
        if L < l:
            # squeeze arrow if length shorter than head length
            A = [X,Y-h]
            B = [X+L,Y+H/2]
            C = [X,Y+H+h]
            head_start = 0
            points = [A, B, C]
        else:
            A = [X,Y]
            B = [X+L-l,Y]
            C = [X+L-l,Y-h]
            D = [X+L,Y+H/2]
            E = [X+L-l,Y+H+h]
            F = [X+L-l,Y+H]
            G = [X,Y+H]
            head_start = L - l # relative to the start of the gene, not absolute coords.
            points = [A, B, C, D, E, F, G]

    elif strand == '-':
        head_start = 0
        if L < l:
            # squeeze arrow if length shorter than head length
            A = [X,Y+H/2]
            B = [X+L,Y-h]
            C = [X+L,Y+H+h]
            head_end = L
            points = [A, B, C]
        else:
            A = [X+L,Y]
            B = [X+l,Y]
            C = [X+l,Y-h]
            D = [X,Y+H/2]
            E = [X+l,Y+H+h]
            F = [X+l,Y+H]
            G = [X+L,Y+H]
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
        
    arrow += "{}\t\t<polygon class=\"{}\" ".format(additional_tabs, gid)
    arrow += "points=\"{}\" fill=\"rgb({})\" ".format(" ".join(points_coords), ",".join([str(val) for val in color]))
    arrow += "fill-opacity=\"1.0\" stroke=\"rgb({})\" ".format(",".join([str(val) for val in color_contour]))
    arrow += "stroke-width=\"{}\" {} />\n".format(str(gene_contour_thickness), category)
    
    # paint domains. Domains on the tip of the arrow should not have corners sticking
    #  out of them
    if only_color_genes:
        domain_list = []
    for domain in domain_list:
        #[X, L, H, domain_accession, (domain_name, domain_description), color, color_contour]
        dX = domain[0]
        dL = domain[1]
        dH = domain[2]
        dacc = domain[3]
        dname = domain[4][0]
        ddesc = domain[4][1]
        dcolor = domain[5]
        dccolour = domain[6]
        
        arrow += additional_tabs + "\t\t<g>\n"
        arrow += "{}\t\t\t<title>{} ({})\n\"{}\"</title>\n".format(additional_tabs, dname, dacc, ddesc)
        
        if strand == "+":
            # calculate how far from head_start we (the horizontal guide at y=Y+internal_domain_margin)
            #  would crash with the slope
            # Using similar triangles:
            collision_x = head_length * (h + internal_domain_margin)
            collision_x /= (h + H/2.0)
            collision_x = round(collision_x)
            
            # either option for x_margin_offset work
            #m = -float(h + H/2)/(head_length) #slope of right line
            #x_margin_offset = (internal_domain_margin*sqrt(1+m*m))/m
            #x_margin_offset = -(x_margin_offset)
            x_margin_offset = internal_domain_margin/sin(pi - atan2(h+H/2.0,-head_length))

            if (dX + dL) < head_start + collision_x - x_margin_offset:
                arrow += "{}\t\t\t<rect class=\"{}\" x=\"{}\" ".format(additional_tabs, dacc, str(X+dX))
                arrow += "y=\"{}\" stroke-linejoin=\"round\" ".format(str(Y + internal_domain_margin))
                arrow += "width=\"{}\" height=\"{}\" ".format(str(dL), str(dH))
                arrow += "fill=\"rgb({})\" stroke=\"rgb({})\" ".format(",".join([str(val) for val in dcolor]), ",".join([str(val) for val in dccolour]))
                arrow += "stroke-width=\"{}\" opacity=\"0.75\" />\n".format(str(domain_contour_thickness))
            else:
                del points[:]
                
                if dX < head_start + collision_x - x_margin_offset:
                    # add points A and B
                    points.append([X + dX, Y + internal_domain_margin])
                    points.append([X + head_start + collision_x - x_margin_offset, Y + internal_domain_margin])
                    
                else:
                    # add point A'
                    start_y_offset = (h + H/2)*(L - x_margin_offset - dX)
                    start_y_offset /= head_length
                    start_y_offset = int(start_y_offset)
                    points.append([X + dX, int(Y + H/2 - start_y_offset)])
                    
                    
                # handle the rightmost part of the domain
                if dX + dL >= head_end - x_margin_offset: # could happen more easily with the scaling
                    points.append([X + head_end - x_margin_offset, int(Y + H/2)]) # right part is a triangle
                else:
                    # add points C and D
                    end_y_offset = (2*h + H)*(L - x_margin_offset - dX - dL)
                    end_y_offset /= 2*head_length
                    end_y_offset = int(end_y_offset)

                    points.append([X + dX + dL, int(Y + H/2 - end_y_offset)])
                    points.append([X + dX + dL, int(Y + H/2 + end_y_offset)])
            
                # handle lower part
                if dX < head_start + collision_x - x_margin_offset:
                    # add points E and F
                    points.append([X + head_start + collision_x - x_margin_offset, Y + H - internal_domain_margin])
                    points.append([X + dX, Y + H - internal_domain_margin])                    
                else:
                    # add point F'
                    points.append([X + dX, int(Y + H/2 + start_y_offset)])
            
                       
                del points_coords[:]
                for point in points:
                    points_coords.append(str(int(point[0])) + "," + str(int(point[1])))
                    
                arrow += "{}\t\t\t<polygon class=\"{}\" ".format(additional_tabs, dacc)
                arrow += "points=\"{}\" stroke-linejoin=\"round\" ".format(" ".join(points_coords))
                arrow += "width=\"{}\" height=\"{}\" ".format(str(dL), str(dH))
                arrow += "fill=\"rgb({})\" ".format(",".join([str(val) for val in dcolor]))
                arrow += "stroke=\"rgb({})\" ".format(",".join([str(val) for val in dccolour]))
                arrow += "stroke-width=\"{}\" opacity=\"0.75\" />\n".format(str(domain_contour_thickness))
            
        # now check other direction
        else:
            # calculate how far from head_start we (the horizontal guide at y=Y+internal_domain_margin)
            #  would crash with the slope
            # Using similar triangles:
            collision_x = head_length * ((H/2) - internal_domain_margin)
            collision_x /= (h + H/2.0)
            collision_x = round(collision_x)
            
            x_margin_offset = round(internal_domain_margin/sin(atan2(h+H/2.0,head_length)))
            
            # nice, blocky domains
            if dX > collision_x + x_margin_offset:
                arrow += "{}\t\t\t<rect class=\"{}\" ".format(additional_tabs, dacc)
                arrow += "x=\"{}\" y=\"{}\" ".format(str(X+dX), str(Y + internal_domain_margin))
                arrow += "stroke-linejoin=\"round\" width=\"{}\" height=\"{}\" ".format(str(dL), str(dH))
                arrow += "fill=\"rgb({})\" ".format(",".join([str(val) for val in dcolor]))
                arrow += "stroke=\"rgb({})\" ".format(",".join([str(val) for val in dccolour]))
                arrow += "stroke-width=\"{}\" opacity=\"0.75\" />\n".format(str(domain_contour_thickness))
            else:
                del points[:]
                
                # handle lefthand side. Regular point or pointy start?
                if dX >= x_margin_offset:
                    start_y_offset = round((h + H/2)*(dX - x_margin_offset)/head_length)
                    points.append([X + dX, Y + H/2 - start_y_offset])
                else:
                    points.append([X + x_margin_offset, Y + H/2])
                    
                    
                # handle middle/end
                if dX + dL < collision_x + x_margin_offset:
                    if head_length != 0:
                        end_y_offset = round((h + H/2)*(dX + dL - x_margin_offset)/head_length)
                    else:
                        end_y_offset = 0
                    points.append([X + dX + dL, Y + H/2 - end_y_offset])
                    points.append([X + dX + dL, Y + H/2 + end_y_offset])
                else:
                    points.append([X + collision_x + x_margin_offset, Y + internal_domain_margin])
                    points.append([X + dX + dL, Y + internal_domain_margin])
                    points.append([X + dX + dL, Y + internal_domain_margin + dH])
                    points.append([X + collision_x + x_margin_offset, Y + internal_domain_margin + dH])
                    
                # last point, if it's not a pointy domain
                if dX >= x_margin_offset:
                    points.append([X + dX, Y + H/2 + start_y_offset])
                       
                del points_coords[:]
                for point in points:
                    points_coords.append(str(int(point[0])) + "," + str(int(point[1])))
                    
                arrow += "{}\t\t\t<polygon class=\"{}\" ".format(additional_tabs, dacc)
                arrow += "points=\"{}\" stroke-linejoin=\"round\" ".format(" ".join(points_coords))
                arrow += "width=\"{}\" height=\"{}\" ".format(str(dL), str(dH))
                arrow += "fill=\"rgb({})\" ".format(",".join([str(val) for val in dcolor]))
                arrow += "stroke=\"rgb({})\" ".format(",".join([str(val) for val in dccolour]))
                arrow += "stroke-width=\"{}\" opacity=\"0.75\" />\n".format(str(domain_contour_thickness))
        
        arrow += additional_tabs + "\t\t</g>\n"
    
    arrow += additional_tabs + "\t</g>\n"

    return arrow


def draw_line(X,Y,L):
    """
    Draw a line below genes
    """
    
    line = "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" style=\"stroke:rgb(70,70,70); stroke-width:{} \"/>\n".format(str(X), str(Y), str(X+L), str(Y), str(stripe_thickness))
    
    return line


def new_color(gene_or_domain):
    # see https://en.wikipedia.org/wiki/HSL_and_HSV
    # and http://stackoverflow.com/a/1586291
    
    h = uniform(0, 1) # all possible colors

    if gene_or_domain == "gene":
        s = uniform(0.15, 0.3)
        v = uniform(0.98, 1.0)
    elif gene_or_domain == "domain":
        s = uniform(0.5, 0.75) # lower: less saturated
        v = uniform(0.7, 0.9) # lower = darker
    else:
        sys.exit("unknown kind of color. Should be 'gene' or 'domain'")
        
    r, g, b = tuple(int(c * 255) for c in hsv_to_rgb(h, s, v))
    
    return [r, g, b]


def SVG(write_html, outputfile, GenBankFile, BGCname, identifiers, \
    color_genes, color_domains, pfam_domain_categories, pfam_info, loci,\
    max_width, domains_color_file, new_color_domains, module_list=None, \
    module_method = 'lda', include_list = False, H=30, h=5, l=12, mX=10, \
    mY=10, scaling=30, absolute_start=0, absolute_end=-1,\
    only_color_genes=False):
    '''
    Create the main SVG document:
        - read pfd file with domain information (pfdFile contains complete path)
        - read GenBank document (GenBankFile contains handle of opened file)
        - record genes, start and stop positions, and strands, and associate domains
        - write the SVG files
    module_method: str, should be either 'lda' or 'stat'
    only_color_genes: bool, only color genes specified in gene_color file
    '''
    # for colors not found in colors_genes and color_domains, we need to generate them from scratch
    new_color_genes = {}
    
    SVG_TEXT = "" # here we keep all the text that will be written down as a file
   

    # --- create SVG header. We have to get max_width first
    # This means that we have to read the gbk file once to know num loci, max_width
    if loci == -1:
        try:
            records = list(SeqIO.parse(GenBankFile, "genbank"))[:1]
        except ValueError:
            print(" Arrower: error while opening gbk", GenBankFile)
            return
        else:
            loci = len(records)
            max_width = 0
            for record in records:
                if len(record) > max_width:
                    max_width = len(record)
    
        
    if absolute_end < 0: # absolute_end == -1 means "the whole region"
        absolute_end = max_width
    else:
        if (absolute_end - absolute_start) < max_width: # user specified something shorter than full region
            max_width = float(absolute_end - absolute_start)
        else: # user specified something bigger than full region. Cropping to max_width
            absolute_end = max_width
            
    max_width /= scaling
            
    if write_html:
        if module_list:
            if module_method == 'lda':
                mod_info = 'Topic {}, probability {}'.format(module_list[0],\
                    module_list[1]) +\
                    ', overlap score {}'.format(module_list[2])
            elif module_method == 'stat':
                if len(mods[0]) == 7:
                    mod_info = (\
                        'Family {}, statistical module {}, '.format(\
                        module_list[-1],module_list[0]) +\
                        'occurrence {}, detection cutoff {:.2e}'.format(\
                        module_list[1], float(module_list[4])))
                elif len(mods[0]) == 8:
                    mod_info = (\
                        'Clan {}, family {}, statistical module {}'.format(\
                        module_list[-1],module_list[-2],module_list[0]) +\
                        ', occurrence {}, detection cutoff {:.2e}'.format(\
                        module_list[1], float(module_list[4])))
                else:
                    mod_info = (\
                        'Statistical module {}, occurrence {}, '.format(\
                        module_list[0],module_list[1]) +\
                        'detection cutoff {:.2e}'.format(float(\
                        module_list[4])))
            else:
                raise SystemExit('\nWrong method for drawing module (SVG)')
            header = "<div><h2>{}</h2></div>\n".format(mod_info)
            header += "\t\t<div title=\"" + mod_info + "\">\n"
        else:
            header = "<div><h1>{}</h1></div>\n".format(BGCname)
            header += "\t\t<div title=\"" + BGCname + "\">\n"
        additional_tabs = "\t\t\t"
        
        header += "{}<svg width=\"{}\" height=\"{}\">\n".format(additional_tabs, str(max_width + 2*(mX)), str(loci*(2*h + H + 2*mY)))

        addY = loci*(2*h + H + 2*mY)
    else:
        header = "<svg version=\"1.1\" baseProfile=\"full\" xmlns=\"http://www.w3.org/2000/svg\" width=\"" + str(max_width + 2*(mX)) + "\" height=\"" + str(loci*(2*h + H + 2*mY)) + "\">\n"
        addY = 0
        
        additional_tabs = "\t"
              
    SVG_TEXT = header
              
    # For info on the color matrix definition: 
    #  https://www.w3.org/TR/SVG11/filters.html#feColorMatrixElement
    # Core Bio: "#DC143C", (220, 20, 60) Dark red
    # Other Bio: 
    #  original: "#DF809D", (223, 128, 157) Pink .87, 0.5, 0.61
    #  alternative: #f4a236, (244,162,54) 0.95, 0.63, 0.21
    # Transporter: "#3F9FBA" (63, 159, 186) Blue
    #  32839a, (50, 131, 154), 0.19, 0.51, 0.6
    # Regulator: "#63BB6D" (99, 187, 109) Green
    #  #127E1B, (18,126,27) 0.07, 0.49, 0.1
    if len(pfam_domain_categories) > 0:
        filters = additional_tabs + "<filter id=\"shadow_CoreBio\" color-interpolation-filters=\"sRGB\" x=\"-65%\" y=\"-25%\" width=\"230%\" height=\"150%\">\n"
        filters += additional_tabs + "\t<feColorMatrix in=\"SourceGraphic\" result=\"matrixOut\" type=\"matrix\" values=\"0 0 0 0 0.85 0 0 0 0 0.08 0 0 0 0 0.23 0 0 0 1 0\" />\n"
        filters += additional_tabs + "\t<feGaussianBlur in=\"matrixOut\" result=\"blurOut\" stdDeviation=\"7\" />\n"
        filters += additional_tabs + "\t<feBlend in=\"SourceGraphic\" in2=\"blurOut\" mode=\"normal\" />\n"
        filters += additional_tabs + "</filter>\n"
        
        filters += additional_tabs + "<filter id=\"shadow_OtherBio\" color-interpolation-filters=\"sRGB\" x=\"-65%\" y=\"-25%\" width=\"230%\" height=\"150%\">\n"
        filters += additional_tabs + "\t<feColorMatrix in=\"SourceGraphic\" result=\"matrixOut\" type=\"matrix\" values=\"0 0 0 0 0.95 0 0 0 0 0.63 0 0 0 0 0.21 0 0 0 1 0\" />\n"
        filters += additional_tabs + "\t<feGaussianBlur in=\"matrixOut\" result=\"blurOut\" stdDeviation=\"7\" />\n"
        filters += additional_tabs + "\t<feBlend in=\"SourceGraphic\" in2=\"blurOut\" mode=\"normal\" />\n"
        filters += additional_tabs + "</filter>\n"
        
        filters += additional_tabs + "<filter id=\"shadow_Transporter\" color-interpolation-filters=\"sRGB\" x=\"-65%\" y=\"-25%\" width=\"230%\" height=\"150%\">\n"
        filters += additional_tabs + "\t<feColorMatrix in=\"SourceGraphic\" result=\"matrixOut\" type=\"matrix\" values=\"0 0 0 0 0.19 0 0 0 0 0.51 0 0 0 0 0.6 0 0 0 1 0\" />\n"
        filters += additional_tabs + "\t<feGaussianBlur in=\"matrixOut\" result=\"blurOut\" stdDeviation=\"7\" />\n"
        filters += additional_tabs + "\t<feBlend in=\"SourceGraphic\" in2=\"blurOut\" mode=\"normal\" />\n"
        filters += additional_tabs + "</filter>\n"
        
        filters += additional_tabs + "<filter id=\"shadow_Regulator\" color-interpolation-filters=\"sRGB\" x=\"-65%\" y=\"-25%\" width=\"230%\" height=\"150%\">\n"
        filters += additional_tabs + "\t<feColorMatrix in=\"SourceGraphic\" result=\"matrixOut\" type=\"matrix\" values=\"0 0 0 0 0.07 0 0 0 0 0.49 0 0 0 0 0.1 0 0 0 1 0\" />\n"
        filters += additional_tabs + "\t<feGaussianBlur in=\"matrixOut\" result=\"blurOut\" stdDeviation=\"7\" />\n"
        filters += additional_tabs + "\t<feBlend in=\"SourceGraphic\" in2=\"blurOut\" mode=\"normal\" />\n"
        filters += additional_tabs + "</filter>\n"
        
        SVG_TEXT += filters

    # --- read in GenBank file

    # handle domains    
    
    loci = 0
    feature_counter = 1
    records = list(SeqIO.parse(GenBankFile, "genbank"))[:1] #list: not crash the loop
    #todo: figure out how to deal with multiple record genbanks
    for seq_record in records:
        add_origin_Y = loci * (2*(h+mY) + H)

        # draw a line that coresponds to cluster size
        ClusterSize = len(seq_record.seq)
        if (absolute_end - absolute_start) < ClusterSize:
            ClusterSize = (absolute_end - absolute_start)
        
        line = draw_line(mX, add_origin_Y + mY + h + H/2, ClusterSize/scaling)
        
        SVG_TEXT += additional_tabs + "<g>\n"
        
        SVG_TEXT += additional_tabs + "\t" + line
        
        # Calculate features for all arrows
        cds_num=1
        for feature in [feature for feature in seq_record.features if feature.location.start >= absolute_start and feature.location.end <= absolute_end]:
            if feature.type == 'CDS':
                # Get name
                try:
                    GeneName = feature.qualifiers['gene'][0]
                    cds_tag = GeneName
                except KeyError:
                    GeneName = 'NoName'
                    cds_tag = ""
                
                if "locus_tag" in feature.qualifiers:
                    cds_tag += " (" + feature.qualifiers["locus_tag"][0] + ")"
                if "product" in feature.qualifiers:
                    cds_tag += "\n" + feature.qualifiers["product"][0]
                    
                # Get color
                # color = (255,255,255)
                # if color_genes:
                    # color = (255,255,255)
                #try:
                    #color = color_genes[GeneName]
                #except KeyError:
                    #color = new_color("gene")
                    #new_color_genes[GeneName] = color
                    #color_genes[GeneName] = color
                    #pass
                
                color_contour = (0,0,0)
                # change to hsv color palette to lower shade for contour color
                #h_, s, v = rgb_to_hsv(float(color[0])/255.0, float(color[1])/255.0, float(color[2])/255.0)
                #color_contour = tuple(int(c * 255) for c in hsv_to_rgb(h_, s, 0.4*v))
                
                # Get strand
                strand = feature.strand
                if strand == -1:
                    strand = '-'
                elif strand == 1:
                    strand = '+'
                else:
                    sys.exit("Weird strand value: " + strand)
                
                # define arrow's start and end
                # http://biopython.org/DIST/docs/api/Bio.SeqFeature.FeatureLocation-class.html#start
                start = feature.location.start - absolute_start
                start = int(start/scaling)
                stop = feature.location.end - absolute_start
                stop = int(stop/scaling)
                
                # assemble identifier to match domains with this feature
                # try:
                    # protein_id = feature.qualifiers['protein_id'][0]
                # except KeyError:
                    # protein_id = ""
                    # pass
                # identifier = BGCname + "_ORF" + str(feature_counter)
                # identifier += ":gid::" if GeneName == "NoName" else ":gid:" + str(GeneName) + ":"
                # identifier += "pid:" + str(protein_id) + ":loc:" + str(feature.location.start) + ":" + str(feature.location.end)
                # identifier = identifier.replace("<","").replace(">","")

                # gene category according to domain content
                #has_core = False
                #has_otherbio = False
                #has_transporter = False
                #has_regulator = False
                #for row in identifiers[identifier]:
                    #dom_acc = row[3]
                    #cat = ""
                    #try:
                        #cat = pfam_domain_categories[dom_acc]
                    #except KeyError:
                        #pass
                    
                    #if cat == "Core Biosynthetic":
                        #has_core = True
                    #if cat == "Other Biosynthetic":
                        #has_otherbio = True
                    #if cat == "Transporter":
                        #has_transporter = True
                    #if cat == "Regulator":
                        #has_regulator = True
                            
                gene_category = ""
                #if has_core:
                    #gene_category = "filter=\"url(#shadow_CoreBio)\""
                #if has_otherbio and not (has_core or has_transporter or has_regulator):
                    #gene_category = "filter=\"url(#shadow_OtherBio)\""
                #if has_transporter and not (has_core or has_otherbio or has_regulator):
                    #gene_category = "filter=\"url(#shadow_Transporter)\""
                #if has_regulator and not (has_core or has_otherbio or has_transporter):
                    #gene_category = "filter=\"url(#shadow_Regulator)\""
                        
                identifier = BGCname+'_'+str(cds_num)

                # Get color
                color = (255,255,255)
                if only_color_genes:
                    color = (255,255,255) #chose other color?
                    try:
                        color = color_genes[identifier]
                    except:
                        pass
                #try:
                    #color = color_genes[GeneName]
                #except KeyError:
                    #color = new_color("gene")
                    #new_color_genes[GeneName] = color
                    #color_genes[GeneName] = color
                    #pass

                #X, Y, L, l, H, h, strand, color, color_contour, category, gid, domain_list
                domain_list = identifiers[identifier]
                should_draw_arrow = True
                if module_list:
                    #get doms in gene_list
                    gene_list = [info[3] for info in domain_list]
                    if include_list:
                        #only include domains from analysis
                        ngene_list = []
                        for dom in gene_list:
                            #check if domain is a subPfam
                            m = re.search(r'_c\d+$',dom)
                            if m:
                                if dom[:m.start()] in include_list:
                                    ngene_list.append(dom)
                            else:
                                if dom in include_list:
                                    ngene_list.append(dom)
                        domains_in_gene = ';'.join(ngene_list)
                    else:
                        domains_in_gene = ';'.join(gene_list)
                    #stat index is 5, lda index is 3
                    mod_i = 3 if module_method == 'lda' else 5
                    if not domains_in_gene in module_list[mod_i]:
                        should_draw_arrow = False
                if should_draw_arrow:
                    arrow = draw_arrow(additional_tabs, start+mX,\
                    add_origin_Y+mY+h,\
                    int(feature.location.end-feature.location.start)/scaling,\
                    l, H, h, strand, color, color_contour, gene_category,\
                    cds_tag, domain_list, only_color_genes)
                    if arrow == "":
                        print("  (ArrowerSVG) Warning: something went wrong with {}".format(BGCname))
                    SVG_TEXT += arrow
                
                feature_counter += 1
                cds_num += 1
        loci += 1
        
        SVG_TEXT += additional_tabs + "</g>\n"

    SVG_TEXT += additional_tabs[:-2] + "</svg>\n"
    
    if write_html:
        SVG_TEXT += "\t\t</div>\n"
    
    # finally append new colors to file:
    #if len(new_color_genes) > 0:
        #if len(new_color_genes) < 10:
            #print("  Saving new color names for genes " + ", ".join(new_color_genes.keys()))
        #else:
            #print("  Saving new color names for 10+ genes...")
            
        #with open(gene_color_file, "a") as color_genes_handle:
            #for new_names in new_color_genes:
                #color_genes_handle.write(new_names + "\t" + ",".join([str(ncg) for ncg in new_color_genes[new_names]]) + "\n")
    
    if len(new_color_domains) > 0:
        #if len(new_color_domains) < 10:
            #print("   Saving new color names for domains " + ", ".join(new_color_domains.keys()))
        #else:
            #print("   Saving new color names for 10+ domains")
            
        with open(domains_color_file, "a") as color_domains_handle:
            for new_names in new_color_domains:
                color_domains_handle.write(new_names + "\t" + ",".join([str(ncdom) for ncdom in new_color_domains[new_names]]) + "\n")

    
    mode = "a" if write_html == True else "w"
    with open(outputfile, mode) as handle:
        handle.write(SVG_TEXT)

def read_dom_hits(dom_hits_file,color_domains,pfam_info,scaling=30,H=30):
    '''Returns dict of {gene_identifier:[[domain_info]]}
    '''
    print('\nReading dom_hits file')
    if not os.path.isfile(dom_hits_file):
        sys.exit("Error (Arrower): " + dom_hits_file + " not found")
    new_color_domains = {}
    identifiers = defaultdict(list)
    with open(dom_hits_file, "r") as pfd_handle:
        pfd_handle.readline() #header
        for line in pfd_handle:
            row = line.strip('\n').split("\t")

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
            #get strand
            loc = row[3].replace("<","").replace(">","")
            g_start,g_end,strand = loc.split(';')

            #get start and end of pfam
            pf_start,pf_end = row[-2].split(';')
            width = 3*(int(pf_end) - int(pf_start))

            if strand == "+":
                # multiply by 3 because the env. coordinate is in aminoacids, not in bp
                # This start is relative to the start of the gene
                start = 3*int(pf_start)
            else:
                loci_start = int(g_start)
                loci_end = int(g_end)
                                
                start = loci_end - loci_start - 3*int(pf_start) - width
            
            # geometry
            start = int(start/scaling)
            width = int(width/scaling)

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
            h_, s, v = rgb_to_hsv(float(color[0])/255.0, float(color[1])/255.0, float(color[2])/255.0)
            color_contour = tuple(int(c * 255) for c in hsv_to_rgb(h_, s, 0.8*v))


            # [X, L, H, domain_acc, color, color_contour]
            identifier = row[0]+'_'+row[4]
            desc = pfam_info.get(domain_acc,('',''))
            identifiers[identifier].append([start, width, int(H - 2*internal_domain_margin), domain_acc, desc, color, color_contour])
    print('  read domains')
    return identifiers,new_color_domains

def read_modules(filename, lda_or_stat='lda'):
    '''Returns dict of {bgc_name:[[info,module_tuple]]}

    lda_or_stat: str, either lda or stat to specify the method
    '''
    mod_dict = defaultdict(list)
    print('\nReading modules ({})'.format(lda_or_stat))
    with open(filename,'r') as inf:
        header = '' #just in case
        for mod in inf:
            if mod.startswith('>'):
                header = mod.strip()[1:]
            else:
                if not mod.startswith('cl') and not mod.startswith('kn'):
                    mod = mod.strip().split('\t')
                    if lda_or_stat == 'lda':
                        mod_tup = tuple([m.split(':')[0] for m in \
                            mod[-1].split(',')])
                        mod_list = mod[:-1] + [mod_tup]
                    elif lda_or_stat == 'stat':
                        mod_tup = tuple(mod[5].split(','))
                        mod_list = mod[:5] + [mod_tup] + mod[6:]
                    else:
                        raise SystemExit(\
                            '\nInvalid method while reading modules, '+
                            'should be either lda or stat')
                    mod_dict[header].append(mod_list)
    return mod_dict

def read_txt(in_file):
    '''Reads text file into list

    in_file: str, file path
    '''
    with open(in_file, 'r') as inf:
        lines = [line.strip() for line in inf]
    return lines


if __name__ == '__main__':
    cmd = get_commands()

    if cmd.modules_lda:
        modules_lda = read_modules(cmd.modules_lda)
    else:
        modules_lda = cmd.modules_lda
    if cmd.modules_stat:
        modules_stat = read_modules(cmd.modules_stat, lda_or_stat='stat')
    else:
        modules_stat = cmd.modules_stat

    with open(cmd.outfile,'w') as outf:
        pass  # clear outfile

    if cmd.one:
        files = [cmd.filenames, ]
    else:
        with open(cmd.filenames,'r') as inf:
            files = [line.strip() for line in inf]

    if cmd.include_list:
        include_doms = read_txt(cmd.include_list)
    else:
        include_doms = False

    domain_colours = read_color_domains_file(cmd.domains_colour_file)
    if cmd.genes_colour_file:
        gene_colours = read_color_genes_file(cmd.genes_colour_file)
        only_colour_genes = True
    else:
        gene_colours = {}
        only_colour_genes = False
    pfam_info = {}

    dom_hits,new_colour_doms = read_dom_hits(cmd.dom_hits_file,domain_colours,\
        pfam_info)

    print('\nVisualising sub-clusters')
    for filename in files:
        if cmd.verbose:
            print(filename)
        bgc = os.path.split(filename)[1].split('.gbk')[0]
        SVG(True, cmd.outfile,filename,bgc,dom_hits,gene_colours,domain_colours,{},\
            pfam_info,-1,None,cmd.domains_colour_file,new_colour_doms,\
            only_color_genes=only_colour_genes)
        if modules_lda:
            for module in modules_lda[bgc]:
                plot=True
                if cmd.topic_include:
                    if not module[0] in cmd.topic_include:
                        plot=False
                if plot:
                    SVG(True, cmd.outfile,filename,bgc,dom_hits,gene_colours,\
                        domain_colours,{}, pfam_info,-1,None,\
                        cmd.domains_colour_file,new_colour_doms,\
                        module_list=module, module_method = 'lda',\
                        include_list=include_doms,\
                        only_color_genes=only_colour_genes)
        if modules_stat:
            mods = modules_stat[bgc]
            if not mods:
                if cmd.verbose:
                    print('\tNo statistical modules present')
                continue
            if len(mods[0]) == 7:
                #sort on family
                mods.sort(key=lambda x: int(x[-1]))
            elif len(mods[0]) == 8:
                #sort on clan and then family
                mods.sort(key=lambda x: (int(x[-1]),int(x[-2])))
            for module in mods:
                plot=True
                if cmd.include_stat_module:
                    if not module[0] in cmd.include_stat_module:
                        plot=False
                if cmd.include_stat_family:
                    if not module[6] in cmd.include_stat_family:
                        plot=False
                if cmd.include_stat_clan:
                    if not module[7] in cmd.include_stat_clan:
                        plot=False
                if plot:
                    SVG(True, cmd.outfile,filename,bgc,dom_hits,gene_colours,\
                        domain_colours,{}, pfam_info,-1,None,\
                        cmd.domains_colour_file,new_colour_doms,\
                        module_list=module, module_method = 'stat',\
                        include_list=include_doms,\
                        only_color_genes=only_colour_genes)

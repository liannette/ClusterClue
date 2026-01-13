from pathlib import Path 
import argparse
import sys
import re

    
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Filters motifs.")
    parser.add_argument(
        "--gene_prob_file", dest="gene_prob_file", required=True, metavar="<file>", 
        help="File containing the subcluster motif gene probabilities.")
    parser.add_argument(
        "--min_gene_prob", dest="min_gene_prob", required=True, metavar="<float>",
        help="Genes need to have at least this probability to be considered \
        part of the subcluster motif.")
    parser.add_argument(
        "--min_core_genes", dest="min_core_genes", required=True, metavar="<int>",
        help="A subcluster motif must have at least this many genes with a probability equal or \
        higher than the minimum core gene probability.")
    parser.add_argument(
        "--core_threshold", dest="core_threshold", required=True, 
        metavar="<float>", help="A gene is considered a subcluster motifcore gene if its \
            probability is at least as high as this value.")
    parser.add_argument(
        "--min_matches", dest="min_matches", required=True, metavar="<int>",
        help="A motif must have occured at least this many times to not be removed.")
    parser.add_argument(
        "--output", dest="filtered_gene_prob_file", required=True, metavar="<file>",
        help="Output file, this will contain the filtered subcluster motifs and their gene \
        probabilities.")
    return parser.parse_args()
    

def passes_min_matches(n_matches, min_matches):
    return True if int(n_matches) >= int(min_matches) else False


def passes_min_core_genes(probabilities, min_core_genes, prob_threshold):
    core_gene_prob = [prob for prob in probabilities if prob >= prob_threshold]
    return True if len(core_gene_prob) >= min_core_genes else False


def remove_low_probabilities(gene_probs, min_probability):
    """
    Removes genes that have a probability lower than min_probability
    """
    filtered_gene_probs = {gene:prob for gene, prob in gene_probs.items() if prob >= min_probability}
    filtered_genes = list(filtered_gene_probs.keys())
    filtered_probabilities = list(filtered_gene_probs.values())
    return filtered_genes, filtered_probabilities

    
def main():
    print("Command-line:", " ".join(sys.argv))
    
    args = parse_arguments()
    gene_prob_file = Path(args.gene_prob_file)
    min_gene_prob = float(args.min_gene_prob)
    min_matches = int(args.min_matches)
    min_core_genes = int(args.min_core_genes)
    core_threshold = float(args.core_threshold)
    filtered_gene_prob_file = Path(args.filtered_gene_prob_file)
    
    # read input file
    with open(gene_prob_file, "r") as infile:
        lines = infile.readlines()
    
    filtered_lines = []

    removed_due_to_low_matches = 0
    removed_due_to_low_core_genes = 0
    removed_due_to_insufficient_gene_number = 0
    
    # iterate though each subcluster
    for i in range(0, len(lines), 3):
        header_line, genes_line, probs_line = lines[i:i+3]
        subcluster_id, n_matches = [int(num) for num in re.findall(r'\d+', header_line)]
        tokenised_genes = genes_line.rstrip().split()
        probabilities = [float(prob) for prob in probs_line.rstrip().split()]

        # check if motif passes filter
        if not passes_min_matches(n_matches, min_matches):
            removed_due_to_low_matches += 1
            print(f"Motif {subcluster_id} removed, it has only {n_matches} matches.")
            continue
        if not passes_min_core_genes(probabilities, min_core_genes, core_threshold):
            removed_due_to_low_core_genes += 1
            print(f"Motif {subcluster_id} removed, it has only {len([p for p in probabilities if p >= core_threshold])} core genes.")
            continue

        # remove probabilities below the min probability threshold
        filtered_genes, filtered_probs = remove_low_probabilities(
            dict(zip(tokenised_genes, probabilities)), 
            min_gene_prob
            )

        if len(filtered_genes) < 2:
            removed_due_to_insufficient_gene_number += 1
            print(f"Motif {subcluster_id} removed, it has only {len(filtered_genes)} genes left after filtering.")
            continue

        # add to filtered subclusters
        genes_line = "\t".join(filtered_genes) + "\n"
        probs_line = "\t".join(map(str, [round(p, 3) for p in filtered_probs])) + "\n"
        filtered_lines.extend([header_line, genes_line, probs_line])
        
    # write to outfile
    with open(filtered_gene_prob_file, "w") as outfile:
        outfile.writelines(filtered_lines)
    
    print("")
    print(f"Removed {removed_due_to_low_matches} motifs due to low matches.")
    print(f"Removed {removed_due_to_low_core_genes} motifs due to low core genes.")
    print(f"Removed {removed_due_to_insufficient_gene_number} motifs due to insufficient number of genes.")

if __name__ == "__main__":
    main()
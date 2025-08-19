from pathlib import Path 
import argparse
import sys
import re
import numpy as np
    
    
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clusters", dest="clusters_file", required=True,
        metavar="<file>", help="File containing the tokenised bgcs")
    parser.add_argument(
        "--weights", dest="weights_file", required=True,
        metavar="<file>", help="File containing the subcluster weights and thresholds")
    parser.add_argument(
        "--output", dest="output_file", required=True, metavar="<file>",
        help="")
    return parser.parse_args()


def parse_clusters_file(clusters_file):
    with open(clusters_file, "r") as infile:
        clusters = [read_cluster(line) for line in infile]
        clusters = {bgc_id: bgc_genes for bgc_id, bgc_genes in clusters}
    return clusters


def read_cluster(line):
    bgc_id, tokenized_genes = line.rstrip().split(",", 1)
    tokenized_genes = set(tokenized_genes.split(","))
    tokenized_genes.discard("-") # genes without biosynthetic domains
    return bgc_id, tokenized_genes


def parse_weights_file(weights_file):
    subcluster_models = dict()
    infile = open(weights_file, "r")
    while True:
        # read 4 lines at a time
        lines = [infile.readline().rstrip() for _ in range(4)]
        # stop it end of file
        if not lines[0]:
            break
        # add subcluster model
        sc_model = GWM.from_lines(lines)
        subcluster_models[sc_model.model_id] = sc_model
    infile.close()
    return subcluster_models


class GWM:
    def __init__(self, model_id, n_matches, threshold, tokenized_genes, weight_matrix):
        self.model_id = model_id
        self.n_matches = n_matches
        self.threshold = float(threshold)
        self.tokenized_genes = tokenized_genes
        self.weight_matrix = weight_matrix
        
    @classmethod
    def from_lines(cls, lines):
        sc_id, n_matches, threshold = re.findall(r"\d+\.\d+|\d+", lines[0])
        tokenised_genes = lines[1].split()
        weights_present = [float(num) for num in lines[2].split()]
        weights_absent = [float(num) for num in lines[3].split()]
        weight_matrix = np.array([weights_present, weights_absent]).transpose()
        return cls(sc_id, n_matches, threshold, tokenised_genes, weight_matrix)
        
    def calculate_score(self, cluster):
        """
        Calculates the subcluster score
        """
        score = sum(
            self.weight_matrix[i][0] if gene in cluster else self.weight_matrix[i][1]
            for i, gene in enumerate(self.tokenized_genes)
            )
        return score


def main(clusters_file, weights_file, output_file):
    
    clusters = parse_clusters_file(clusters_file)
    gwms = parse_weights_file(weights_file)
    
    results = []
    for bgc_id, bgc_genes in clusters.items():
        for gwm in gwms.values():
            score = gwm.calculate_score(bgc_genes)
            if score < gwm.threshold:
                continue
            common_genes = set(gwm.tokenized_genes) & bgc_genes
            if len(common_genes) < 2:
                continue
            results.append([bgc_id, gwm, score, common_genes]) 

    with open(output_file, "w") as outfile:
        # print header
        print("\t".join(["cluster", "model", "n_matches", "threshold", "score", "tokenised_genes"]), 
              file=outfile)
        # print results
        for bgc_id, gwm, score, genes in results:
            threshold = str(round(gwm.threshold, 3))
            score = str(round(score, 3))
            genes = ",".join(sorted(genes))
            print("\t".join([bgc_id, gwm.model_id, gwm.n_matches, threshold, score, genes]), 
                  file=outfile)


if __name__ == "__main__":

    print("Command-line:", " ".join(sys.argv))
    
    args = parse_arguments()
    clusters_file = Path(args.clusters_file)
    weights_file = Path(args.weights_file)
    output_file = Path(args.output_file)

    main(clusters_file, weights_file, output_file)
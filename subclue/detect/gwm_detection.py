import re
import numpy as np
import csv

from collections import defaultdict


def parse_domain_hits_file(domain_hits_file_path):
    domains_in_bgc = defaultdict(lambda: defaultdict(set))
    with open(domain_hits_file_path, "r") as f:
        csv_reader = csv.DictReader(f, delimiter="\t")
        for row in csv_reader:
            bgc_id = row["bgc"]
            protein_id = row["p_id"]
            domain = row["domain"]
            domains_in_bgc[bgc_id][protein_id].add(domain)
    return domains_in_bgc


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
        
    def detect(self, domain_hits):
        score = 0
        hit_protein_ids = set()

        for i, gene in enumerate(self.tokenized_genes):
            domains = gene.split(";")
            
            # Check if any orf contains all domains
            protein_with_all_domains = None
            for p_id, protein_domains in domain_hits.items():
                if all(d in protein_domains for d in domains):
                    protein_with_all_domains = p_id
                    break
            
            # Update score based on presence of matching protein
            if protein_with_all_domains:
                score += self.weight_matrix[i][0]
                hit_protein_ids.add(protein_with_all_domains)  # Add hit protein ID immediately
            else:
                score += self.weight_matrix[i][1]

        return score, hit_protein_ids




def main(domain_hits_file_path, weights_file, output_file, verbose):

    gwms = parse_weights_file(weights_file)
    bgcs = parse_domain_hits_file(domain_hits_file_path)
    
    results = []
    for bgc_id, domain_hits in bgcs.items():
        for gwm in gwms.values():
            score, protein_ids = gwm.detect(domain_hits)
            if score < gwm.threshold:
                continue
            if len(protein_ids) < 2:
                continue
            results.append([bgc_id, gwm, score, protein_ids])

    with open(output_file, "w") as outfile:
        # print header
        print("\t".join(["bgc_id", "model_id", "model_n_matches", "model_score_threshold", "hit_score", "hit_protein_ids"]), 
              file=outfile)
        # print results
        for bgc_id, gwm, score, protein_ids in results:
            threshold = str(round(gwm.threshold, 3))
            score = str(round(score, 3))
            protein_ids = ",".join(sorted(protein_ids))
            print("\t".join([bgc_id, gwm.model_id, gwm.n_matches, threshold, score, protein_ids]), 
                  file=outfile)

    if verbose:
        print(f"Detected {len(results)} motifs in {len(bgcs)} BGCs using {len(gwms)} GWM models.")

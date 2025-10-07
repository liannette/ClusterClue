import re
import numpy as np


def parse_clusters_file(clusters_file):
    with open(clusters_file, "r") as infile:
        clusters = [read_cluster(line) for line in infile]
        clusters = {bgc_id: bgc_genes for bgc_id, bgc_genes in clusters}
    return clusters


def read_cluster(line):
    bgc_id, tokenized_genes = line.rstrip().split(",", 1)
    tokenized_genes = set(tokenized_genes.split(","))
    tokenized_genes.discard("-")  # genes without biosynthetic domains
    return bgc_id, tokenized_genes


def parse_motifs_file(motifs_file):
    subcluster_motifs = dict()
    infile = open(motifs_file, "r")
    while True:
        # read 4 lines at a time
        lines = [infile.readline().rstrip() for _ in range(4)]
        # stop it end of file
        if not lines[0]:
            break
        # add subcluster motif
        motif = Motif.from_lines(lines)
        subcluster_motifs[motif.motif_id] = motif
    infile.close()
    return subcluster_motifs


class Motif:
    def __init__(self, motif_id, n_matches, threshold, tokenized_genes, weight_matrix):
        self.motif_id = motif_id
        self.n_matches = n_matches
        self.threshold = float(threshold)
        self.tokenized_genes = tokenized_genes
        self.weight_matrix = weight_matrix

    @classmethod
    def from_lines(cls, lines):
        motif_id, n_matches, threshold = re.findall(r"\d+\.\d+|\d+", lines[0])
        tokenised_genes = lines[1].split()
        weights_present = [float(num) for num in lines[2].split()]
        weights_absent = [float(num) for num in lines[3].split()]
        weight_matrix = np.array([weights_present, weights_absent]).transpose()
        return cls(motif_id, n_matches, threshold, tokenised_genes, weight_matrix)

    def calculate_score(self, cluster):
        """
        Calculates the subcluster score
        """
        score = sum(
            self.weight_matrix[i][0] if gene in cluster else self.weight_matrix[i][1]
            for i, gene in enumerate(self.tokenized_genes)
        )
        return score


def main(clusters_file, weights_file, output_file, verbose):

    if verbose:
        print(f"\nDetecting motifs in {clusters_file} using weights from {weights_file}")

    clusters = parse_clusters_file(clusters_file)
    motifs = parse_motifs_file(weights_file)

    if verbose:
        print(f"Parsed {len(clusters)} clusters and {len(motifs)} motifs.")

    results = []
    for bgc_id, bgc_genes in clusters.items():
        for motif in motifs.values():
            score = motif.calculate_score(bgc_genes)
            if score < motif.threshold:
                continue
            common_genes = set(motif.tokenized_genes) & bgc_genes
            if len(common_genes) < 2:
                continue
            results.append([bgc_id, motif, score, common_genes])

    if verbose:
        print(f"Detected {len(results)} motifs across {len(clusters)} clusters.")

    with open(output_file, "w") as outfile:
        # print header
        header_fields = [
            "bgc_id",
            "motif_id",
            "n_training",
            "score_threshold",
            "score",
            "genes",
        ]
        print("\t".join(header_fields), file=outfile)
        # print lines
        for bgc_id, motif, score, common_genes in results:
            threshold = str(round(motif.threshold, 3))
            score = str(round(score, 3))
            common_genes = ",".join(sorted(common_genes))
            line_fields = [
                bgc_id,
                motif.motif_id,
                motif.n_matches,
                threshold,
                score,
                common_genes,
            ]
            print("\t".join(line_fields), file=outfile)

    if verbose:
        print(f"Detected motifs written to {output_file}")

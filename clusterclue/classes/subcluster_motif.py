import numpy as np
from collections import Counter
import re


class SubclusterMotif:
    def __init__(self, 
        motif_id, 
        matches=None,
        n_matches=None,
        tokenized_genes=None,
        probabilities=None, 
        gwm=None,
        threshold=None
        ):
        self.motif_id = motif_id
        self.matches = matches
        if matches is None:
            self.n_matches = n_matches
        else:
            self.n_matches = len(matches)
        self.tokenized_genes = tokenized_genes
        self.probabilities = probabilities
        self.gwm = gwm
        self.threshold = threshold


    @classmethod
    def from_lines(cls, lines):
        motif_id = re.search(r"motif: (\w+)", lines[0]).group(1)
        n_matches = re.search(r"n_matches: (\d+)", lines[0]).group(1)
        threshold = re.search(r"threshold: (-?[\d.]+)", lines[0]).group(1)
        tokenized_genes = lines[1].split()
        weights_present = [float(num) for num in lines[2].split()]
        weights_absent = [float(num) for num in lines[3].split()]
        weight_matrix = np.array([weights_present, weights_absent]).transpose()
        return cls(
            motif_id,
            n_matches=int(n_matches),
            tokenized_genes=tokenized_genes,
            gwm=weight_matrix,
            threshold=float(threshold)
            )


    def calculate_gene_probabilities(self, min_prob=0.001):

        gene_probs = []
        # count gene occurrences across matches
        gene_counter = Counter()
        for bgc_id, module in self.matches.items():
            gene_counter.update(set(module))
        
        # calculate probabilities and filter by min_prob to reduce size
        for gene, count in gene_counter.items():
            probability = count / self.n_matches
            if probability >= min_prob:
                gene_probs.append((gene, probability))
        
        # sort genes by probability
        gene_probs.sort(key=lambda x: x[1], reverse=True)

        self.tokenized_genes = [gene for gene, _ in gene_probs]
        self.probabilities = [probability for _, probability in gene_probs]
        

    def calculate_score(self, cluster):
        """
        Calculates the subcluster score
        """
        score = sum(
            self.gwm[i][0] if gene in cluster else self.gwm[i][1]
            for i, gene in enumerate(self.tokenized_genes)
            )
        return round(score, 3)

    def _filter_low_probabilities(self, min_prob):
        """Removes genes with probabilities below min_prob from the motif."""
        filtered_genes = []
        filtered_probs = []
        for gene, prob in zip(self.tokenized_genes, self.probabilities):
            if prob >= min_prob:
                filtered_genes.append(gene)
                filtered_probs.append(prob)
        self.tokenized_genes = filtered_genes
        self.probabilities = filtered_probs

    def _calculate_threshold(self):
        """Calculates the threshold for the motif. The threshold is the lowest 
        score among all matches that were used to generate the gwm."""
        matches = self.matches.values()
        scores = [self.calculate_score(match) for match in matches]
        threshold = min(scores)
        return threshold

    def _calculate_weight_matrix(self, background_counts, n_clusters):
        """Calculates the weight matrix for the motif."""
        def create_probability_matrix(probabilities):
            """Creates a probability matrix for the tokenised genes"""
            prob_matrix = []
            for prob in probabilities:
                prob_matrix.append([prob, 1-prob])
            return np.array(prob_matrix)

        prob_matrix = create_probability_matrix(self.probabilities)
        bg_probs = [background_counts[g]/n_clusters for g in self.tokenized_genes]
        bg_prob_matrix = create_probability_matrix(bg_probs)

        with np.errstate(divide='ignore'): # ignore log2(0) warnings
            weight_matrix = np.log2(prob_matrix / bg_prob_matrix)

        # round to 3 decimal places
        return np.round(weight_matrix, decimals=3)

    def generate_gwm_with_threshold(self, background_counts, n_clusters):
        """Generates the GWM and threshold for the motif. The threshold is 
        calculated based on the matches used to generate the GWM. The lowest
        score among these matches is used as the threshold.
        """
        self.gwm = self._calculate_weight_matrix(background_counts, n_clusters)
        self.threshold = self._calculate_threshold()

    def gwm_to_txt(self):
        motif_id = self.motif_id
        n_matches = self.n_matches
        threshold = self.threshold
        tokenized_genes = self.tokenized_genes
        weights_present, weights_absent = self.gwm.transpose() 
        lines = [
            f"#motif: {motif_id}, n_matches: {n_matches}, threshold: {threshold}", 
            "\t".join(tokenized_genes), 
            "\t".join(map(str, weights_present)), 
            "\t".join(map(str, weights_absent))
            ]
        text = "\n".join(lines) + "\n"
        return text

import numpy as np


class SubclusterMotif:
    def __init__(self, motif_id, n_matches, tokenised_genes, probabilities):
        self.motif_id = motif_id
        self.n_matches = n_matches
        self.tokenised_genes = tokenised_genes
        self.probabilities = probabilities
        self.gwm = None
    
    def gwm_to_txt(self):
        header_line = f"#motif: {self.motif_id}, n_matches: {self.n_matches}"
        tokenised_genes_line = "\t".join(self.tokenised_genes)
        w_present, w_absent = self.gwm.transpose() 
        weights_present_line = "\t".join(map(str, np.round(w_present, decimals=3)))
        weights_absent_line = "\t".join(map(str, np.round(w_absent, decimals=3)))
        lines = [
            header_line, 
            tokenised_genes_line, 
            weights_present_line, 
            weights_absent_line
            ]
        text = "\n".join(lines) + "\n"
        return text
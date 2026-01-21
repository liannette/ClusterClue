class StatModule:
    def __init__(
        self,
        module_id,
        strictest_pval,
        tokenised_genes,
        n_occurrences=None, 
    ):
        self.module_id = module_id
        self.n_genes = len(tokenised_genes)
        self.n_domains = sum(len(domains) for domains in tokenised_genes)
        self.tokenised_genes = tokenised_genes
        self.strictest_pval = strictest_pval
        self.n_occurrences = n_occurrences

    def __repr__(self):
        return (
            f"StatModule("
            f"module_id={self.module_id}, "
            f"strictest_pval={self.strictest_pval}, "
            f"n_genes={self.n_genes}, "
            f"n_domains={self.n_domains}, "
            f"tokenised_genes={self.tokenised_genes}), "
            f"n_occurrences={self.n_occurrences})"
        )

    def __eq__(self, other):
        if not isinstance(other, StatModule):
            return False
        return self.tokenised_genes == other.tokenised_genes

    def __hash__(self):
        # Convert tokenised_genes to a tuple of tuples to make it hashable
        tokenised_genes_hashable = tuple(tuple(gene) for gene in self.tokenised_genes)
        return hash(tokenised_genes_hashable)

    def to_dict(self):
        """
        Converts the StatModule object to a dictionary representation.

        Returns:
            dict: A dictionary representation of the StatModule object.
        """
        return {
            "module_id": self.module_id,
            "n_occurrences": self.n_occurrences,
            "strictest_pval": self.strictest_pval,
            "n_genes": self.n_genes,
            "n_domains": self.n_domains,
            "tokenised_genes": self.tokenised_genes,
        }

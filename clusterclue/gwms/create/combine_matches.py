import logging


logger = logging.getLogger(__name__)
        
def read_stat_subcluster_matches(input_filepath):
    module_matches = set()
    with open(input_filepath, "r") as infile:
        for line in infile:
            # header lines contain bgc_id
            if line.startswith(">"):
                bgc_id = line.rstrip()[1:]
            else:
                module_id, module = line.rstrip().split()
                tokenised_genes = tuple(sorted(module.split(",")))
                module_matches.add((bgc_id, tokenised_genes))
    return module_matches


def read_top_subcluster_matches(input_filepath):
    """
    Returns a dict with the subcluster module (tokenised genes) as key 
    and bgc IDs as value
    """
    module_matches = set()
    with open(input_filepath, "r") as infile:
        for line in infile:
            # skip header lines
            if line.startswith("#"):
                continue

            _, _, match, bgc_id, _ = line.rstrip().split()

            tokenised_genes = set()
            for gene_and_topic_prob in match.split(","):
                gene, _ = gene_and_topic_prob.split(":")
                tokenised_genes.add(gene)

            # must have at least 2 unique genes
            if len(tokenised_genes) >= 2:
                module_matches.add((bgc_id, tuple(sorted(tokenised_genes))))
    return module_matches


def combine_presto_matches(stat_matches_filepath, top_matches_filepath, output_filepath):

    # Load clusterclue subcluster predictions
    stat_matches = read_stat_subcluster_matches(stat_matches_filepath)
    top_matches = read_top_subcluster_matches(top_matches_filepath)
    combined_matches = stat_matches | top_matches

    # Write combined matches to file
    with open(output_filepath, "w") as outfile:
        for bgc_id, mod in sorted(combined_matches):
            outfile.write(f"{bgc_id}\t{','.join(mod)}\n")
    
    logger.info(
        f"Wrote {len(combined_matches)} ({len(stat_matches)} STAT + "
        f"{len(top_matches)} TOP) presto subcluster predictions to {output_filepath}"
        )

    return combined_matches

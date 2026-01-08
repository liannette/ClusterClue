import logging


logger = logging.getLogger(__name__)
        
def read_stat_subcluster_matches(input_filepath):
    module_matches = []
    with open(input_filepath, "r") as infile:
        for line in infile:
            # header lines contain bgc_id
            if line.startswith(">"):
                bgc_id = line.rstrip()[1:]
            else:
                module_id, module = line.rstrip().split()
                module = module.split(",")  # Split modules by comma
                tokenised_genes = tuple(sorted(module))
                module_matches.append([bgc_id, tokenised_genes])
    return module_matches


def _get_genes_from_top_match_line(line):
    genes = set()
    match = line.rstrip().split()[2]
    for gene_and_topic_prob in match.split(","):
        gene, _ = gene_and_topic_prob.split(":")
        genes.add(gene)
    return tuple(sorted(genes))


def read_top_subcluster_matches(input_filepath):
    """
    Returns a dict with the subcluster module (tokenised genes) as key 
    and bgc IDs as value
    """
    module_matches = []
    with open(input_filepath, "r") as infile:
        for line in infile:
            # skip header lines
            if line.startswith("#"):
                continue
            bgc_id = line.split()[3]
            tokenised_genes = _get_genes_from_top_match_line(line)
            # must have at least 2 unique genes
            if len(tokenised_genes) >= 2:
                module_matches.append([bgc_id, tokenised_genes])
    return module_matches


def write_combined_matches(matches, output_filepath):
    matches.sort(key=lambda x: x[0])
    with open(output_filepath, "w") as outfile:
        for bgc_id, mod in matches:
            outfile.write(f"{bgc_id}\t{','.join(mod)}\n")


def combine_matches(stat_matches_filepath, top_matches_filepath, out_filepath):

    # Load stat subcluster predictions
    stat_matches = read_stat_subcluster_matches(stat_matches_filepath)
    logger.info(f"Loaded {len(stat_matches)} predicted STAT subclusters")

    # Load top subcluster predictions   
    top_matches = read_top_subcluster_matches(top_matches_filepath)
    logger.info(f"Loaded {len(top_matches)} predicted TOP subclusters")

    # Combine
    combined_matches = stat_matches + top_matches
    logger.info(f"Total predicted subclusters: {len(combined_matches)}")
    write_combined_matches(combined_matches, out_filepath)
    logger.info(f"Wrote combined matches to {out_filepath}")

    return combined_matches

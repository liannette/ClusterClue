import csv
import logging
from collections import OrderedDict
from ipresto.clusters.utils import count_non_empty_genes
from ipresto.presto_stat.stat_module import StatModule

logger = logging.getLogger(__name__)

#TODO: make sure that the outfiles are sorted: modules_per_bgc, stat_module_k_families

def read_clusterfile(clusterfile, m_gens, verbose):
    """
    Reads a cluster file into a dictionary of {bgc: [(domains_of_a_gene)]}.

    Parameters:
        clusterfile (str): The file path to the input cluster file.
        m_gens (int): The minimum number of genes with domains a cluster should have.
        verbose (bool): If True, print additional information during processing.

    Returns:
        dict: A dictionary where keys are cluster names (bgc) and values are lists of tuples,
            each tuple representing the protein domains in a gene.

    The function also prints the number of clusters read and the number of clusters excluded
    due to having fewer genes than the specified minimum. Clusters with non-unique names
    will trigger a warning message.

    Notes:
    - Clusters with fewer than `m_gens` genes are not included in the returned dictionary.
    - The function also prints the number of clusters excluded due to having fewer genes than `m_gens`.
    - Domains represented by '-' are not counted in the gene count.
    """
    logger.info(f"Reading {clusterfile}")
    with open(clusterfile, "r") as inf:
        clusters = OrderedDict()
        for line in inf:
            line = line.strip().split(",")
            clus = line[0]
            genes = line[1:]
            g_doms = [tuple(gene.split(";")) for gene in genes]
            if clus not in clusters.keys():
                clusters[clus] = g_doms
            else:
                logger.warning(f"Duplicate cluster ID {clus}.")

    # remove clusters with too few genes
    if m_gens > 0:
        clusters = remove_empty_clusters(clusters, m_gens, verbose)

    logger.info(f"Read {len(clusters)} clusters")
    return clusters


def remove_empty_clusters(clusters, min_genes, verbose):
    """
    Removes clusters with fewer than `min_genes` genes from the clusters dictionary.

    Parameters:
        clusters (dict): The dictionary of clusters to filter.
        min_genes (int): The minimum number of genes with domains a cluster should have.
        verbose (bool): If True, print additional information during processing.

    Returns:
        dict: The filtered dictionary of clusters, where each cluster has at least `min_genes` genes.

    """
    to_delete = []
    for clus, genes in clusters.items():
        n_genes = count_non_empty_genes(genes)
        if n_genes < min_genes:
            if verbose:
                logger.debug(
                    f"  Excluding {clus}: contains only {n_genes} genes with "
                    f"domain hits."
                )
            to_delete.append(clus)

    for clus in to_delete:
        del clusters[clus]

    if to_delete:
        logger.info(
            f"{len(to_delete)} clusters excluded for having less than "
            f"{min_genes} genes with domain hits (minimum required: {min_genes})."
        )

    return clusters


def tokenized_genes_to_string(tokenized_genes):
    """
    Converts a list of tokenized genes (tuples of tuples) into a string representation.

    Parameters:
        tokenized_genes (list of tuples): A list of tokenized genes, where each gene is
            represented as a tuple of tuples.

    Returns:
        str: A string representation of the tokenized genes, with each gene separated by a comma.
    """
    return ",".join([";".join(gene) for gene in tokenized_genes])


def string_to_tokenized_genes(genes_string):
    """
    Converts a string representation of tokenized genes back into a list of tuples.

    Parameters:
        genes_string (str): A string representation of tokenized genes, where each gene is
            separated by a comma and each tuple is separated by a semicolon.

    Returns:
        list of tuples: A list of tokenized genes, where each gene is represented as a tuple of tuples.
    """
    return [tuple(gene.split(";")) for gene in genes_string.split(",")]


def write_stat_modules(modules: dict, file_path: str):
    """
    Writes the statistical modules to a file in tab-separated format.

    Parameters:
        modules (dict): A dictionary where keys are module IDs and values are StatModule objects.
        file_path (str): The file path to the output file.

    The function writes the header based on the keys of the first dictionary in the list,
    and then writes each module's values in tab-separated format.
    """
    if not modules:
        logger.warning("No modules to write.")
        return

    header = list(modules.values())[0].to_dict().keys()
    with open(file_path, "w", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=header, delimiter="\t")
        writer.writeheader()
        for module in modules.values():
            row = module.to_dict()
            row["tokenised_genes"] = tokenized_genes_to_string(row["tokenised_genes"])
            writer.writerow(row)


def read_stat_modules(file_path):
    """
    Reads statistical modules from a tab-separated file into a list of StatModules.

    Parameters:
        file_path (str): The file path to the input file.

    Returns:
        dict: A dictionary where keys are module IDs and values are StatModule objects.
    """
    with open(file_path, "r", newline="") as infile:
        reader = csv.DictReader(infile, delimiter="\t")
        modules = {}
        for row in reader:
            module_id = row["module_id"]
            strictest_pval = float(row["strictest_pval"])
            tokenised_genes = string_to_tokenized_genes(row["tokenised_genes"])
            if module_id in modules:
                logger.warning(f"Duplicate module ID {module_id}. Keeping the first one.")
                continue
            module = StatModule(
                module_id=module_id,
                strictest_pval=strictest_pval,
                tokenised_genes=tokenised_genes,
            )
            modules[module_id] = module
        logger.info(f"Read {len(modules)} STAT modules.")
        return modules


def write_modules_per_bgc(modules_per_bgc, out_file_path):
    """
    Writes the modules per BGC to a file in tab-separated format.

    Parameters:
        modules_per_bgc (dict): A dictionary where keys are BGC IDs and values are sets of module IDs.
        out_file_path (str): The file path to the output file.
    """
    with open(out_file_path, "w", newline="") as outfile:
        writer = csv.writer(outfile, delimiter="\t")
        header = ["bgc_id", "module_ids"]
        writer.writerow(header)
        for bgc_id, module_ids in modules_per_bgc.items():
            writer.writerow([bgc_id, ",".join(sorted(module_ids))])


def write_bgcs_per_module(bgcs_per_module, out_file_path):
    """
    Writes the BGCs per module to a file in tab-separated format.

    Parameters:
        bgcs_per_module (dict): A dictionary where each key is a module ID and each value is a
            set of associated BGC IDS.
        out_file_path (str): The file path to the output file.
    """
    with open(out_file_path, "w", newline="") as outfile:
        writer = csv.writer(outfile, delimiter="\t")
        header = ["module_id", "bgc_ids"]
        writer.writerow(header)
        for module_id, bgc_ids in bgcs_per_module.items():
            writer.writerow([module_id, ",".join(sorted(bgc_ids))])


def write_detected_stat_modules(modules_per_bgc, modules, out_file_path):
    with open(out_file_path, "w", newline="") as outfile:
        writer = csv.writer(outfile, delimiter="\t")
        for bgc_id, module_ids in modules_per_bgc.items():
            outfile.write(f">{bgc_id}\n")
            for module_id in module_ids:
                module = modules[module_id]
                tokenised_genes = tokenized_genes_to_string(module.tokenised_genes)
                writer.writerow(
                    [
                        module_id,
                        tokenised_genes,
                    ]
                )

def write_module_families(module_ids, family_ids, out_file_path):
    with open(out_file_path, "w", newline="") as outfile:
        writer = csv.writer(outfile, delimiter="\t")
        header = ["module_id", "module_family_id"]
        writer.writerow(header)
        for module_id, family_id in zip(module_ids, family_ids):
            writer.writerow([module_id, family_id])

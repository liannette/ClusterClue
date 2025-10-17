import warnings
# Suppress UserWarning from NumPy when importing statsmodels about smallest 
# subnormal float values being zero, caused by underlying native libraries 
# enabling flush-to-zero mode for performance. This warning does not affect 
# numerical accuracy for our use case (statistical p-value corrections)
# because values like p-values are never close to subnormal float32 limits.
# Suppression keeps logs clean without hiding real issues.
warnings.filterwarnings("ignore", message="The value of the smallest subnormal for")
import logging
import networkx as nx
from collections import Counter, defaultdict, OrderedDict
from itertools import product
from multiprocessing import Pool
from functools import partial
from statsmodels.stats.multitest import multipletests
from sympy import binomial as ncr
from math import floor, log10
from ipresto.presto_stat.detect import detect_modules_in_bgcs, get_bgcs_per_module
from ipresto.presto_stat.stat_module import StatModule

logger = logging.getLogger(__name__)


def makehash():
    """Function to initialise nested dict"""
    return defaultdict(makehash)


def count_adj(counts, cluster):
    """Counts all adjacency interactions between domains in a cluster

    counts: nested dict { dom1:{ count:x,N1:y,N2:z,B1:{dom2:v},B2:{dom2:w} } }
    cluster: list of tuples, genes with domains
    """
    if len(cluster) == 1:
        return
    for i, dom in enumerate(cluster):
        if i == 0:
            edge = 1
            adj = [cluster[1]]
        elif i == len(cluster) - 1:
            edge = 1
            adj = [cluster[i - 1]]
        else:
            edge = 2
            adj = [cluster[i - 1], cluster[i + 1]]
            if adj[0] == adj[1] and adj[0] != ("-",):
                # B2 and N2 counts
                prevdom = cluster[i - 1]
                counts[prevdom]["N1"] -= 2
                counts[prevdom]["N2"] += 1
                if dom != ("-",) and dom != prevdom:
                    counts[prevdom]["B1"][dom] -= 2
                    try:
                        counts[prevdom]["B2"][dom] += 1
                    except TypeError:
                        counts[prevdom]["B2"][dom] = 1
        if not dom == ("-",):
            counts[dom]["count"] += 1
            counts[dom]["N1"] += edge
            for ad in adj:
                if ad != ("-",) and ad != dom:
                    try:
                        counts[dom]["B1"][ad] += 1
                    except TypeError:
                        counts[dom]["B1"][ad] = 1


def remove_dupl_doms(cluster):
    """
    Replaces duplicate domains in a cluster with '-', writes domain at the end

    cluster: list of tuples, tuples contain str domain names
    """
    domc = Counter(cluster)
    dupl = [dom for dom in domc if domc[dom] > 1 if not dom == ("-",)]
    if dupl:
        newclus = [("-",) if dom in dupl else dom for dom in cluster]
        for dom in dupl:
            newclus += [("-",), dom]
    else:
        newclus = cluster
    return newclus


def count_coloc(counts, cluster):
    """Counts all colocalisation interactions between domains in a cluster

    counts: nested dict { dom1:{ count:x,N1:y,B1:{dom2:v,dom3:w } } }
    cluster: list of tuples, genes with domains
    verbose: bool, if True print additional info
    """
    N1 = len(cluster) - 1
    for dom in cluster:
        if not dom == ("-",):
            counts[dom]["count"] += 1
            counts[dom]["N1"] += N1
            coloc = set(cluster)
            try:
                coloc.remove(("-",))
            except KeyError:
                pass
            coloc.remove(dom)
            for colo in coloc:
                try:
                    counts[dom]["B1"][colo] += 1
                except TypeError:
                    counts[dom]["B1"][colo] = 1


def count_interactions(clusdict, verbose):
    """Count all adj and coloc interactions between all domains in clusdict

    clusdict: dict of {cluster:[(gene_with_domains)]}
    verbose: bool, if True print additional info
    Returns two dicts, one dict with adj counts and one with coloc counts
    adj counts:
        { dom1:{ count:x,N1:y,N2:z,B1:{dom2:v},B2:{dom2:w} } }
    coloc counts:
        { dom1:{ count:x,N1:y,B1:{dom2:v,dom3:w } } }
    """
    logger.info("Counting colocalisation and adjacency interactions")
    all_doms = {v for val in clusdict.values() for v in val}
    if ("-",) in all_doms:
        all_doms.remove(("-",))

    # initialising count dicts
    adj_counts = makehash()
    for d in all_doms:
        for v in ["count", "N1", "N2"]:
            adj_counts[d][v] = 0
        for w in ["B1", "B2"]:
            adj_counts[d][w] = makehash()
        # N1: positions adj to one domA, N2: positions adj to two domA
        # B1: amount of domB adj to one domA, B2: positions adj to two domA

    coloc_counts = makehash()
    for d in all_doms:
        for v in ["count", "N1"]:
            coloc_counts[d][v] = 0
        coloc_counts[d]["B1"] = makehash()
        # N1: all possible coloc positions in a cluster, cluster lenght - 1
        # B1: amount of domB coloc with domA

    for clus in clusdict.values():
        count_adj(adj_counts, clus)
        filt_clus = remove_dupl_doms(clus)
        count_coloc(coloc_counts, filt_clus)
    return adj_counts, coloc_counts


def calc_adj_pval(domval_pair, counts, Nall):
    """Returns a list of sorted tuples (domA,domB,pval)

    domval_pair: tuple of (domA, {count:x,N1:y,N2:z,B1:{dom2:v},B2:{dom2:w}} )
    Nall: int, all possible positions
    counts: nested dict { domA:{ count:x,N1:y,N2:z,B1:{dom2:v},B2:{dom2:w} } }
    """
    domA, vals = domval_pair
    # domains without interactions do not end up in pvals
    if not vals["B1"] and not vals["B2"]:
        return
    pvals = []
    count = vals["count"]
    Ntot = Nall - count
    N1 = vals["N1"]
    N2 = vals["N2"]
    N0 = Ntot - N1 - N2
    interactions = vals["B1"].keys() | vals["B2"].keys()
    for domB in interactions:
        if domB not in vals["B2"]:
            B1 = vals["B1"][domB]
            Btot = counts[domB]["count"]
            pval = float(
                1
                - sum([ncr(N0, (Btot - d)) * ncr(N1, d) for d in range(B1)])
                / ncr(Ntot, Btot)
            )
        elif vals["B1"][domB] == 0:
            B2 = vals["B2"][domB]
            Btot = counts[domB]["count"]
            pval = float(
                1
                - sum([ncr(N0, (Btot - d)) * ncr(N2, d) for d in range(B2)])
                / ncr(Ntot, Btot)
            )
        else:
            B1 = vals["B1"][domB]
            B2 = vals["B2"][domB]
            Btot = counts[domB]["count"]
            pval = float(
                1
                - sum(
                    [
                        ncr(N0, Btot - d1 - d2) * ncr(N1, d1) * ncr(N2, d2)
                        for d1, d2 in product(range(B1 + 1), range(B2 + 1))
                        if d1 + d2 != B1 + B2
                    ]
                )
                / ncr(Ntot, Btot)
            )
        ab_int = sorted((domA, domB))
        pvals.append((ab_int[0], ab_int[1], pval))
    return pvals


def calc_adj_pval_wrapper(count_dict, clusdict, cores, verbose):
    """Returns list of tuples of corrected pvals for each gene pair

    counts: nested dict { dom1:{ count:x,N1:y,N2:z,B1:{dom2:v},B2:{dom2:w} } }
    clusdict: dict of {cluster:[(domains_in_a_gene)]}
    cores: int, amount of cores to use
    verbose: bool, if True print additional information
    """
    logger.info("Calculating adjacency p-values")
    N = sum([len(values) for values in clusdict.values()])
    pool = Pool(cores, maxtasksperchild=5)
    pvals_ori = pool.map(
        partial(calc_adj_pval, counts=count_dict, Nall=N), count_dict.items()
    )
    pool.close()
    pool.join()
    # remove Nones, unlist and sort
    pvals_ori = [lst for lst in pvals_ori if lst]
    pvals_ori = sorted([tup for lst in pvals_ori for tup in lst])
    # to check if there are indeed 2 pvalues for each combination
    check_ps = [(tup[0], tup[1]) for tup in pvals_ori]
    check_c = Counter(check_ps)
    pvals = [p for p in pvals_ori if check_c[(p[0], p[1])] == 2]
    if not len(pvals) == len(pvals_ori):
        p_excl = [p for p in pvals if check_c[(p[0], p[1])] != 2]
        excluded_pairs = [f"{p[0]}-{p[1]}" for p in p_excl]
        logger.info(f"Found {len(excluded_pairs)} domain pairs with inconsistent p-values")
        logger.info(f"Excluded domain pairs: {', '.join(excluded_pairs)}")
    # Benjamini-Yekutieli multiple testing correction
    pvals_adj = multipletests(list(zip(*pvals))[2], method="fdr_by")[1]
    # adding adjusted pvals and choosing max
    ptups = []
    for ab1, ab2, p1, p2 in zip(
        pvals[::2], pvals[1::2], pvals_adj[::2], pvals_adj[1::2]
    ):
        assert ab1[0] == ab2[0] and ab1[1] == ab2[1]
        pmax = max([p1, p2])
        ptups.append(((ab1[0], ab1[1]), pmax))
    return ptups


def calc_coloc_pval(domval_pair, counts, Nall):
    """Returns a list of sorted tuples (domA,domB,pval)

    domval_pair: tuple of (domA, { count:x,N1:y,B1:{dom2:v,dom3:w } })
    counts: nested dict { domA:{ count:x,N1:y,B1:{dom2:v,dom3:w } } }
    Nall: int, all possible positions in all clusters
    """
    domA, vals = domval_pair
    # domains without interactions do not end up in pvals
    if not vals["B1"]:
        return
    pvals = []
    count = vals["count"]
    Ntot = Nall - count
    N1 = vals["N1"]
    N0 = Ntot - N1
    interactions = vals["B1"].keys()
    for domB in interactions:
        B1 = vals["B1"][domB]
        Btot = counts[domB]["count"]
        pval = float(
            1
            - sum([ncr(N0, (Btot - d)) * ncr(N1, d) for d in range(B1)])
            / ncr(Ntot, Btot)
        )
        ab_int = sorted((domA, domB))
        pvals.append((ab_int[0], ab_int[1], pval))
    return pvals


def calc_coloc_pval_wrapper(count_dict, clusdict, cores, verbose):
    """Returns list of tuples of corrected pvals for each domain pair

    counts: nested dict { domA:{ count:x,N1:y,B1:{dom2:v,dom3:w } } }
    clusdict: dict of {cluster:[domains]}
    cores: int, amount of cores to use
    verbose: bool, if True print additional information
    """
    logger.info("Calculating colocalisation p-values")
    N = sum([len(remove_dupl_doms(values)) for values in clusdict.values()])
    pool = Pool(cores, maxtasksperchild=1)
    pvals_ori = pool.map(
        partial(calc_coloc_pval, counts=count_dict, Nall=N), count_dict.items()
    )
    pool.close()
    pool.join()
    # remove Nones, unlist and sort
    pvals_ori = [lst for lst in pvals_ori if lst]
    pvals_ori = sorted([tup for lst in pvals_ori for tup in lst])
    # to check if there are indeed 2 pvalues for each combination
    check_ps = [(tup[0], tup[1]) for tup in pvals_ori]
    check_c = Counter(check_ps)
    pvals = [p for p in pvals_ori if check_c[(p[0], p[1])] == 2]
    if not len(pvals) == len(pvals_ori):
        excluded_pairs = [p for p in pvals if check_c[(p[0], p[1])] != 2]
        logger.info(f"Found {len(excluded_pairs)} domain pairs with inconsistent p-values")
        logger.info(f"Excluded domain pairs: {', '.join(excluded_pairs)}")
    # Benjamini-Yekutieli multiple testing correction
    pvals_adj = multipletests(list(zip(*pvals))[2], method="fdr_by")[1]
    # adding adjusted pvals and choosing max
    ptups = []
    for ab1, ab2, p1, p2 in zip(
        pvals[::2], pvals[1::2], pvals_adj[::2], pvals_adj[1::2]
    ):
        assert ab1[0] == ab2[0] and ab1[1] == ab2[1]
        pmax = max([p1, p2])
        ptups.append(((ab1[0], ab1[1]), pmax))
    return ptups


def keep_lowest_pval(colocs, adjs):
    """
    Returns all domain pairs with their lowest pvalue as an edge for nx

    colocs, adjs: list of tuples [((dom1,dom2),pval)]
    Tuples look like (dom1,dom2,{pval:x})
    """
    pvals = colocs + adjs
    counter = Counter(list(zip(*pvals))[0])
    dupl = sorted([tup for tup in pvals if counter[tup[0]] == 2])
    uniques = [tup for tup in pvals if counter[tup[0]] == 1]
    lowest = []
    for p1, p2 in zip(dupl[::2], dupl[1::2]):
        pmin = min([p1[1], p2[1]])
        lowest.append((p1[0][0], p1[0][1], {"pval": pmin}))
    uniques = [(tup[0][0], tup[0][1], {"pval": tup[1]}) for tup in uniques]
    return lowest + uniques


def generate_graph(edges, verbose):
    """Returns a networkx graph

    edges: list/generator of tuples, (pair1,pair2,{attributes})
    """
    g = nx.Graph()
    g.add_edges_from(edges)
    logger.info(f"Generated graph with {g.number_of_nodes()} nodes "
                f"and {g.number_of_edges()} edges")
    return g


def find_modules_by_pval_cutoff(pval_cutoff, edges):
    """
    Returns modules found given a specific pval cutoff as (pval_cutoff, {modules})

    pval_cutoff: float, cutoff for detecting modules
    gene_pairs: list of tuples, ('gene1', 'gene2', {"pval": pvalue})
    Modules are all maximal cliques with length > 2
    """
    filtered_edges = (e for e in edges if e[2]["pval"] <= pval_cutoff)
    module_graph = generate_graph(filtered_edges, False)
    cliques = nx.algorithms.clique.find_cliques(module_graph)
    modules = {tuple(sorted(module)) for module in cliques if len(module) > 2}
    return pval_cutoff, modules


def round_to_n(x, n):
    """Round x to n significant decimals

    x: int/float
    n: int
    """
    if x <= 0:
        return 0
    return round(x, -int(floor(log10(x))) + (n - 1))


def identify_significant_modules(edges, max_pval, cores, verbose):
    """
    Generates a dictionary of modules with their strictest p-value cutoff.

    This function identifies all modules with a p-value lower than the specified
    cutoff and returns a dictionary where the keys are the modules and the values
    are the strictest p-value cutoff for each module.

    Args:
        edges (list of tuples): A list of tuples containing two genes and their
            associated interaction p-values: ('gene1', 'gene2', {"pval": pvalue}))
        max_pval (float): The p-value cutoff for significance.
        cores (int): The number of CPU cores to use for parallel processing.
        verbose (bool): If True, prints additional information during execution.

    Returns:
        dict: A dictionary where keys are module IDs and values are StatModule objects.
    """
    logger.info(
            f"Identifying subcluster modules applying a maximum interaction "
            f"p-value cutoff of {max_pval}"
        )

    significant_edges = [e for e in edges if e[2]["pval"] <= max_pval]
    logger.info(f"Found {len(significant_edges)} significant gene pair interactions")

    pval_cutoffs = {pv["pval"] for pv in list(zip(*significant_edges))[2]}
    if len(pval_cutoffs) > 100000:  # reduce the number of pvals to loop through
        pval_cutoffs = {round_to_n(x, 3) for x in pval_cutoffs}
    logger.info(f"Looping through {len(pval_cutoffs)} p-value cutoffs...")

    pool = Pool(cores, maxtasksperchild=10)
    results = pool.imap(
        partial(find_modules_by_pval_cutoff, edges=significant_edges),
        pval_cutoffs,
        chunksize=250,
    )
    pool.close()
    pool.join()

    modules = {}
    module_id = 1
    for result in results:
        pval_cutoff, generated_modules = result
        for mod in generated_modules:
            if mod not in modules:
                modules[mod] = StatModule(
                    module_id=module_id,
                    strictest_pval=pval_cutoff,
                    tokenised_genes=mod,
                )
                module_id += 1
            else:
                strictest_pval = min(modules[mod].strictest_pval, pval_cutoff)
                modules[mod].strictest_pval = strictest_pval

    logger.info(f"Identified {len(modules)} significant modules.")

    return {mod.module_id: mod for mod in modules.values()}


def calculate_interaction_pvals(
    bgcs,
    cores,
    verbose,
):
    """
    Returns:
        p_values (list of tuples): A list of tuples containing two genes and their
                associated interaction p-values: ('gene1', 'gene2', {"pval": pvalue}))
    """
    adj_counts, c_counts = count_interactions(bgcs, verbose)
    adj_pvals = calc_adj_pval_wrapper(adj_counts, bgcs, cores, verbose)
    col_pvals = calc_coloc_pval_wrapper(c_counts, bgcs, cores, verbose)
    pvals = keep_lowest_pval(col_pvals, adj_pvals)
    # todo: keep from crashing when there are no significant modules
    return pvals


def generate_stat_modules(
    bgcs,
    max_pval,
    cores,
    verbose,
):
    # find the modules
    p_values = calculate_interaction_pvals(bgcs, cores, verbose)
    modules = identify_significant_modules(p_values, max_pval, cores, verbose)

    # remove modules that do not occur in the bgc dataset
    logger.info("Removing modules that do not occur at least twice in the BGC dataset")
    min_occurence = 2
    existing_modules = filter_infrequent_modules(
        modules, bgcs, min_occurence, cores
    )

    # report how many modules were removed
    n_removed = len(modules) - len(existing_modules)
    percent_removed = n_removed / len(modules) * 100
    logger.info(f"Removed {n_removed} ({round(percent_removed, 2)}%) modules.")

    # remove modules that are contained by another one, occuring the same amount in the dataset
    logger.info("Removing modules that are contained by another one")
    filtered_modules = filter_redundant_modules(existing_modules, bgcs, cores)

    # report how many modules were removed
    n_removed = len(existing_modules) - len(filtered_modules)
    percent_removed = n_removed / len(modules) * 100
    logger.info(f"Removed {n_removed} ({round(percent_removed, 2)}%) modules.")

    return filtered_modules


def filter_infrequent_modules(
    modules: dict, bgcs: dict, min_occurence: int, cores: int
) -> dict:
    """Removes modules that occur less than min_occurence in the BGC dataset.

    Args:
        modules (dict): A dictionary where keys are module IDs and values are StatModule objects.
        bgcs (dict): A dictionary where each key is a BGC identifier and each value is a list of genes.
            Each gene is a tuple of protein domains.
        min_occurence (int): The minimum number of occurrences for a module.
        cores (int): The number of CPU cores to use for parallel processing.

    Returns:
        dict: A dictionary of modules that occur at least min_occurence times in the BGC dataset.


    Note:
        The module IDs in the returned list are renumbered starting from 1.

    """
    modules_per_bgc = detect_modules_in_bgcs(bgcs, modules, cores)
    bgcs_per_module = get_bgcs_per_module(modules, modules_per_bgc)

    new_id = 1
    updated_modules = OrderedDict()
    for mod_id, mod in modules.items():
        n_occurences = len(bgcs_per_module[mod_id])
        if n_occurences >= min_occurence:
            new_module = StatModule(
                module_id=mod_id,
                strictest_pval=mod.strictest_pval,
                tokenised_genes=mod.tokenised_genes,
                n_occurrences=n_occurences,
            )
            updated_modules[new_id] = new_module
            new_id += 1

    return updated_modules


def module_is_redundant(module_id: int, modules: dict) -> bool:
    """
    Check if a module is contained in any other module with the same number of occurrences.

    Args:
        module_id (str): The ID of the module to check.
        modules (dict): Dictionary where keys are module IDs and values are StatModule objects.
    Returns:
        bool: True if the module is contained in another, False otherwise.
    """
    gene_set = set(modules[module_id].tokenised_genes)
    for other_module_id, other_module in modules.items():
        if module_id != other_module_id:
            other_gene_set = set(other_module.tokenised_genes)
            if (
                modules[module_id].n_occurrences == modules[other_module_id].n_occurrences and 
                gene_set.issubset(other_gene_set)
                ):
                return True
    return False


def filter_redundant_modules(modules: dict, bgcs: dict, cores: int) -> dict:
    """Removes modules that are contained by another one, occuring the same amount in the dataset.

    Args:
        modules (dict): A dictionary where keys are module IDs and values are StatModule objects.
        bgcs (dict): A dictionary where each key is a BGC identifier and each value is a list of genes.
            Each gene is a tuple of protein domains.
        cores (int): The number of CPU cores to use for parallel processing.

    Returns:
        dict: A dictionary of filtered modules.
    """
    module_ids = list(modules.keys())
  
    pool = Pool(cores, maxtasksperchild=5)
    results = pool.map(
        partial(module_is_redundant, modules=modules), 
        module_ids
    )
    pool.close()
    pool.join()
    is_redundant = dict(zip(module_ids, results))

    new_id = 1
    updated_modules = OrderedDict()
    for mod_id, mod in modules.items():
        if is_redundant[mod_id]:
            continue
        updated_modules[new_id] = StatModule(
            module_id=new_id,
            strictest_pval=mod.strictest_pval,
            tokenised_genes=mod.tokenised_genes,
            n_occurrences=mod.n_occurrences,
        )
        new_id += 1
    return updated_modules

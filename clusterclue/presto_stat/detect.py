from functools import partial
from multiprocessing import Pool


def map_modules_to_bgc(bgc_id: str, bgc_genes: list, modules: dict):
    """
    Returns a tuple of (bgc, [modules]) where each module is a list of genes.

    Parameters:
    bgc_id (str): The name of the biosynthetic gene cluster (BGC).
    bgc_genes (list of tuples): A list of genes, where each gene is a tuple of protein domains.
    modules (dict): A dictionary where keys are module IDs and values are StatModule objects.

    Returns:
    tuple: A tuple containing the BGC id and a set of module ids that are contained in the BGC.
    """
    contained_module_ids = set()
    for mod_id, mod in modules.items():
        if set(mod.tokenised_genes).issubset(set(bgc_genes)):
            contained_module_ids.add(mod_id)
    return bgc_id, contained_module_ids


def detect_modules_in_bgcs(bgcs: dict, modules: dict, cores: int):
    """Detects STAT subcluster modules in tokenized bgcs.

    Args:
        bgcs (dict): A dictionary where each key is a BGC identifiers and each value
            is a list of genes. Each gene is a tuple of protein domains.
        modules (dict): A dictionary where keys are module IDs and values are StatModule objects.
        cores (int): The number of CPU cores to use for parallel processing.

    Returns:
        dict: A dictionary where each key is a BGC ID and each value is a set of associated
            module IDs. 
    """
    # todo: change the modules to the dict structure
    pool = Pool(cores, maxtasksperchild=10)
    results = pool.starmap(partial(map_modules_to_bgc, modules=modules), bgcs.items())
    pool.close()
    pool.join()

    return {bgc_id: contained_module_ids for bgc_id, contained_module_ids in results}


def get_bgcs_per_module(modules: dict, modules_per_bgc: dict) -> dict:
    """Detects BGCs for STAT subcluster modules.

    Args:
        modules (dict): A dictionary where keys are module IDs and values are StatModule objects.
        modules_per_bgc (dict): A dictionary where each key is a BGC identifier and each value
            is a list of module identifiers.

    Returns:
        dict: A dictionary where each key is a module ID and each value is a
            set of associated BGC IDS.
    """
    # for each module, get the BGCs that contain it
    bgcs_per_module = {}
    for mod_id, mod in modules.items():
        bgcs_per_module[mod_id] = set()
        for bgc_id, contained_module_ids in modules_per_bgc.items():
            if mod_id in contained_module_ids:
                bgcs_per_module[mod_id].add(bgc_id)

    return bgcs_per_module

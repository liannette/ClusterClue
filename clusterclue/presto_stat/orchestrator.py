import logging
from pathlib import Path
from typing import Optional
from clusterclue.presto_stat.utils import (
    read_clusterfile,
    read_stat_modules,
    write_modules_per_bgc,
    write_bgcs_per_module,
    write_detected_stat_modules,
)
from clusterclue.presto_stat.generate import generate_stat_modules
from clusterclue.presto_stat.detect import (
    detect_modules_in_bgcs,
    get_bgcs_per_module,
)
from clusterclue.presto_stat.cluster_modules import (
    cluster_stat_modules,
    plot_kmeans_elbow,
)
from clusterclue.presto_stat.utils import write_stat_modules

logger = logging.getLogger(__name__)


class StatOrchestrator:
    def run(
        self,
        out_dir_path: str,
        cluster_file: str,
        stat_modules_file_path: Optional[str],
        pval_cutoff: float,
        n_families_range: list,
        min_genes_per_bgc: int,
        cores: int,
        verbose: bool,
    ) -> str:
        """ """
        # Create output directory
        out_dir = Path(out_dir_path)
        out_dir_path.mkdir(parents=True, exist_ok=True)

        # Read clusters
        bgcs = read_clusterfile(cluster_file, min_genes_per_bgc, verbose)

        # Generate subcluster modules
        if not stat_modules_file_path:
            logger.info(
                "Generating new PRESTO-STAT subcluster modules using the  "
                f"clusters from {cluster_file}"
            )
            modules = generate_stat_modules(
                bgcs,
                pval_cutoff,
                cores,
                verbose,
            )
            stat_modules_file_path = out_dir / "stat_modules.txt"
            write_stat_modules(modules, stat_modules_file_path)
            logger.info(f"PRESTO-STAT subcluster modules have been saved to: {stat_modules_file_path}")

        logger.info(f"Reading PRESTO-STAT subcluster modules from: {stat_modules_file_path}")
        modules = read_stat_modules(stat_modules_file_path)

        # Detect modules in bgcs
        logger.info("Detecting PRESTO-STAT subcluster modules in the input clusters")
        modules_per_bgc = detect_modules_in_bgcs(bgcs, modules, cores)
        write_modules_per_bgc(modules_per_bgc, out_dir / "modules_per_bgc.txt")
        bgcs_per_module = get_bgcs_per_module(modules, modules_per_bgc)
        write_bgcs_per_module(bgcs_per_module, out_dir / "bgcs_per_module.txt")
        write_detected_stat_modules(
            modules_per_bgc, modules, out_dir / "detected_stat_modules.txt"
        )
        logger.info(
            "Detected PRESTO-STAT subcluster modules have been saved to: "
            f"{out_dir / 'detected_stat_modules.txt'}"
        )

        # Cluster the modules into families
        if n_families_range:
            logger.info("Clustering the STAT modules into families.")

            # TODO: automatic decide on best k 
            inertias = cluster_stat_modules(
                modules, n_families_range, cores, out_dir, verbose
            )

            if len(n_families_range) > 1:
                plot_path = out_dir / "stat_modules_families_elbow_plot.png"
                plot_kmeans_elbow(n_families_range, inertias, plot_path)
                logger.info(f"Elbow plot saved to: {plot_path}")

        logger.info("PRESTO-STAT finished.")




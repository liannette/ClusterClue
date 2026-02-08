import logging
from pathlib import Path
from typing import Optional
from clusterclue.presto_top.presto_top import (
    read2dict,
    run_lda,
    run_lda_from_existing,
    process_lda,
    plot_convergence,
)

logger = logging.getLogger(__name__)


def setup_gensim_logger(log_file: Path):
    """Set up a separate logger for gensim to log into a specific file."""
    gensim_logger = logging.getLogger("gensim")
    gensim_logger.setLevel(logging.INFO)
    # Disable propagation so gensim logs aren't sent to root logger
    gensim_logger.propagate = False
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file) for h in gensim_logger.handlers):
        gensim_handler = logging.FileHandler(log_file)
        gensim_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")
        gensim_handler.setFormatter(formatter)
        gensim_logger.addHandler(gensim_handler)


class TopOrchestrator:
    def run(
        self,
        out_dir_path: str,
        cluster_file: str,
        top_model_file_path: Optional[str],
        topics: int, 
        amplify: Optional[int], 
        iterations: Optional[int] = 500,
        chunksize: Optional[int] = None, 
        update: bool = False, 
        visualise: bool = False,
        alpha: str = "symmetric",
        beta: str = "symmetric",
        plot: bool = False,
        feat_num: int = 75,
        min_feat_score: float = 0.95,
        cores: int = 1,
    ) -> str:
        """ """
        # Create output directory
        presto_top_dir = Path(out_dir_path)
        presto_top_dir.mkdir(parents=True, exist_ok=True)

        # separate logging for gensim to a specific log file in the presto_top output dir
        gensim_log_file = presto_top_dir / 'log_presto_top.txt'
        setup_gensim_logger(gensim_log_file)

        # Read clusters
        bgcs = read2dict(cluster_file)

        if amplify:
            bgc_items = []
            for bgc in bgcs.items():
                bgc_items += [bgc] * amplify
            bgclist, dom_list = zip(*bgc_items)
        else:
            bgclist, dom_list = zip(*bgcs.items())

        # Deprecated parameters, keep for compatibility
        bgc_classes_dict = {bgc: 'None' for bgc in bgcs}
        known_subclusters = False

        if top_model_file_path:
            logger.info(f"Using provided model: {top_model_file_path}")
            with open(gensim_log_file, 'w') as outf:
                outf.write(f'\nUsing model from {top_model_file_path}')

            lda, lda_dict, bow_corpus = run_lda_from_existing(
                top_model_file_path, dom_list, presto_top_dir,
                no_below=1, no_above=0.5)
        else:
            logger.info(
                f"Computing new PRESTO-TOP model from the input clusters: "
                f"{cluster_file}"
            )
            logger.info(
                f"Parameters: {topics} topics, {amplify} amplification, "
                f"{iterations} iterations of chunksize {chunksize}"
            )
            lda, lda_dict, bow_corpus = run_lda(
                dom_list, no_below=1, no_above=0.5,
                num_topics=topics, cores=cores, outfolder=presto_top_dir,
                iters=iterations, chnksize=chunksize,
                update_model=update, ldavis=visualise, alpha=alpha,
                beta=beta)


        process_lda(lda, lda_dict, bow_corpus, feat_num, bgcs,
                    min_feat_score, bgclist, presto_top_dir, bgc_classes_dict,
                    num_topics=topics, amplif=amplify, plot=plot,
                    known_subcl=known_subclusters)

        if not top_model_file_path:
            outfile = presto_top_dir / "model_convergence_likelihood.pdf"
            plot_convergence(gensim_log_file, iterations, outfile)
            
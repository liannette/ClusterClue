from pathlib import Path
from typing import Optional
import logging
from ipresto.presto_top.presto_top import (
    read2dict,
    run_lda,
    run_lda_from_existing,
    process_lda,
    plot_convergence,
)


class TopOrchestrator:
    def run(
        self,
        out_dir_path: str,
        cluster_file: str,
        top_model_file_path: Optional[str],
        topics: int, 
        amplify: Optional[int], 
        iterations: int,
        chunksize: int, 
        update: bool, 
        visualise: bool,
        alpha: str,
        beta: str,
        plot: bool,
        feat_num: int,
        min_feat_score: float,
        cores: int,
        verbose: bool,
    ) -> str:
        """ """
        # Create output directory
        presto_top_dir = Path(out_dir_path)
        presto_top_dir.mkdir(parents=True, exist_ok=True)

        # writing detailed log information to seperate log file
        log_out = presto_top_dir / 'log_presto_top.txt'
        logging.basicConfig(filename=log_out,
                            format="%(asctime)s:%(levelname)s:%(message)s",
                            level=logging.INFO)

        # Read clusters
        bgcs = read2dict(cluster_file)

        if amplify:
            bgc_items = []
            for bgc in bgcs.items():
                bgc_items += [bgc] * amplify
            bgclist, dom_list = zip(*bgc_items)
        else:
            bgclist, dom_list = zip(*bgcs.items())


        # if classes:
        #     bgc_classes_dict = read2dict(classes, sep='\t', header=True)
        # else:
        #     bgc_classes_dict = {bgc: 'None' for bgc in bgcs}
        bgc_classes_dict = {bgc: 'None' for bgc in bgcs}

        # if known_subclusters:
        #     known_subclusters = defaultdict(list)
        #     with open(known_subclusters, 'r') as inf:
        #         for line in inf:
        #             line = line.strip().split('\t')
        #             known_subclusters[line[0]].append(line[1:])
        # else:
        #     known_subclusters = False
        known_subclusters = False

        if top_model_file_path:
            if verbose:
                print(
                    f"\nUsing provided model from: {top_model_file_path}"
                )
            with open(log_out, 'w') as outf:
                outf.write(f'\nUsing model from {top_model_file_path}')

            lda, lda_dict, bow_corpus = run_lda_from_existing(
                top_model_file_path, dom_list, presto_top_dir,
                no_below=1, no_above=0.5)
        else:
            if verbose:
                print(
                    "\nComputing new PRESTO-TOP model from the input clusters: "
                    f"{cluster_file}"
                )
                print(
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
            plot_convergence(log_out, iterations, outfile)

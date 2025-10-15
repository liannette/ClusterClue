#!/usr/bin/env python3
"""
Author: Joris Louwen

Script to perform PRESTO-TOP method within iPRESTO.
It finds sub-cluster motifs from a clusterfile with the LDA algorithm.

Usage: presto_top.py -h

Example usage:
python3 presto_top.py -i my_clusterfile.csv -o my_output_folder -c 10
        -t 1000 -C 3000 -I 2000 --min_genes 2 -f 0.95 -n 75 --classes
        my_bgc_classes.txt --known_subclusters known_subcl.txt
"""
import logging
import os
# to account for a weird bug with ldamulticore and numpy:
# https://github.com/RaRe-Technologies/gensim/issues/1988
os.environ['OMP_NUM_THREADS'] = '1'
import matplotlib
matplotlib.use('Agg')  # to not rely on X-forwarding (not available in screen)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim
import re
import scipy.cluster.hierarchy as sch
import seaborn as sns
from collections import Counter, defaultdict
from gensim.models.ldamulticore import LdaMulticore
# from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from gensim.corpora.mmcorpus import MmCorpus
from math import ceil
from numpy import sqrt
from operator import itemgetter
from pathlib import Path

logger = logging.getLogger(__name__)


def run_lda(domlist, no_below, no_above, num_topics, cores, outfolder,
            iters, chnksize, update_model=False, ldavis=True,
            alpha='symmetric', beta=None):
    """
    Returns LDA model with the Dictionary and the corpus, LDAvis is optional

    domlist: list of list of str, list of the bgc domain-combinations
    no_below: int, domain-combinations that occur in less than no_below
        bgcs will be removed
    no_above: float, remove domain-combinations that occur in more than
        no_above fraction of the dataset
    num_topics: int, number of topics
    cores: int, number of cores to use
    outfolder: str, filepath
    iters: int, number of iterations for training
    chnksize: int, chunksize for one training batch
    update_model: bool, update the model (with same parameters) or not
    ldavis: bool, if true save LDAvis visualisation of model
    alpha: str, alpha parameter for lda in string format, either symmetric,
        auto or an integer.
    beta: str, beta parameter for lda in string format, either symmetric,
        auto or an integer.
    """
    outfolder = Path(outfolder)
    model_filename = 'lda_model' 
    model_filepath = outfolder / model_filename

    #model = os.path.join(outfolder, 'lda_model')

    # save the token ids the model will be build on.
    dict_filepath = outfolder / f'{model_filename}.dict'
    if not dict_filepath.exists():
        dict_lda = Dictionary(domlist)
        dict_lda.filter_extremes(no_below=no_below, no_above=no_above)
        dict_lda.save(dict_filepath)
    else:
        logger.info("Loaded existing dict_file with words")
        dict_lda = Dictionary.load(dict_filepath)
    logger.info('Constructing LDA model with {} BGCs and:'.format(len(domlist)),
          dict_lda)
    
    corpus_bow = [dict_lda.doc2bow(doms) for doms in domlist]
    corpus_filepath = outfolder / f'{model_filename}mm'
    if not corpus_filepath.exists():
        MmCorpus.serialize(corpus_filepath, corpus_bow)

    # to allow for x iterations of chunksize y
    passes = ceil(iters * chnksize / len(domlist))
    # gamma_threshold based on Blei et al. 2010
    offst = 1

    # convert alpha/beta in int if needed
    if alpha not in ['symmetric', 'asymmetric', 'auto']:
        alpha = int(alpha)
    if beta not in ['symmetric', 'asymmetric', 'auto']:
        if beta == 'None':
            beta = None
        else:
            beta = int(beta)

    if not model_filepath.exists():
        lda = LdaMulticore(
            corpus=corpus_bow, num_topics=num_topics,
            id2word=dict_lda, workers=cores, per_word_topics=True,
            chunksize=chnksize, iterations=iters, gamma_threshold=0.0001,
            offset=offst, passes=passes, dtype=np.float64, alpha=alpha,
            eta=beta)
        lda.save(model_filepath)
    else:
        logger.info('Loaded existing LDA model')
        lda = LdaMulticore.load(model_filepath)
        if update_model:
            # update the model. to be functional the input should be stationary
            # (no topic drift in new documents)
            logger.info("Existing model is updated")
            # for the multicore model new parameters cannot be added, the
            # parameters from the existing model will be used to update
            lda.update(corpus_bow, chunks_as_numpy=True)
            lda.save(model_filepath)

    if ldavis:
        visname = os.path.join(outfolder, 'lda_method-tsne.html')
        logger.info('Running pyLDAvis for visualisation')
        vis = pyLDAvis.gensim.prepare(
            lda, corpus_bow, dict_lda, sort_topics=False, mds='tsne')
        logger.info('  saving visualisation with t-sne to html')
        try:
            pyLDAvis.save_html(vis, visname)
        except TypeError as e:
            logger.error('  saving visualisation failed with: {}'.format(e))
    return lda, dict_lda, corpus_bow


def run_lda_from_existing(existing_model, domlist, outfolder,
                          no_below=1, no_above=0.5):
    """
    Returns existing LDA model with the Dictionary and the corpus.

    existing_model: str, filepath to lda model
    domlist: list of list of str, list of the bgc domain-combinations
    outfolder: str, filepath
    no_below: int, domain-combinations that occur in less than no_below
        bgcs will be removed
    no_above: float, remove domain-combinations that occur in more than
        no_above fraction of the dataset
    """
    model = existing_model
    # load the token ids the model is build on.
    dict_file = existing_model + '.dict'
    dict_lda = Dictionary.load(dict_file)

    corpus_bow = [dict_lda.doc2bow(doms) for doms in domlist]
    # save current corpus
    corpus_file = os.path.join(outfolder, 'current_corpus.mm')
    if not os.path.isfile(corpus_file):
        MmCorpus.serialize(corpus_file, corpus_bow)

    lda = LdaMulticore.load(existing_model)
    logger.info('Loaded existing LDA model')
    logger.info('Applying existing LDA model on {} BGCs with'.format(len(domlist)),
          dict_lda)
    # cm = CoherenceModel(model=lda, corpus=corpus_bow, dictionary=dict_lda,\
    # coherence='c_v', texts=domlist)
    # coherence = cm.get_coherence()
    # logger.info('Coherence: {}, num_topics: {}'.format(coherence, num_topics))

    return lda, dict_lda, corpus_bow


def process_lda(lda, dict_lda, corpus_bow, feat_num, bgc_dict, min_f_score,
                bgcs, outfolder, bgc_classes, num_topics, amplif=False,
                min_t_match=0.05, min_feat_match=0.3, plot=True,
                known_subcl=False, overlap_score_cutoff=0.15,
                total_feat_score_cutoff=1.1):
    """Analyses the topics in the bgcs

    lda: LdaMulticore, the lda model
    dict_lda
    bgc_dict: dict of {bgc:[domain_combinations]}
    bgcs: (amplified) list of bgc names
    overlap_score_cutoff: float, minimal fraction of the topic that should be
        present in the topic (subcluster) match
    total_feat_score_cutoff: float, minimal amount of features (based on sum
        of their feature scores) that should be present in subcluster, default:
        1.1, meaning that as a proxy for gene presence, that at least more than
        one gene should be present in the subcluster match
    """
    # this is a list of tuple (topic_num, 'features_with_scores')
    lda_topics = lda.print_topics(-1, 75)
    topic_num = len(lda_topics)
    # get the topic names from the lda html visualisation
    ldahtml = os.path.join(outfolder, 'lda.html')
    if os.path.isfile(ldahtml):
        with open(ldahtml, 'r') as inf:
            for line in inf:
                if line.startswith('var lda'):
                    lst_str = line.strip().split('"topic.order": ')[-1]
                    nums = map(int, lst_str.strip('[]};').split(', '))
                    trans = {i_lda - 1: i_vis + 1 for i_vis, i_lda in
                             zip(range(topic_num), nums)}
    else:
        trans = {x: '-' for x in range(topic_num)}
    filt_features, feat_scores, zero_topics = select_number_of_features(
        lda_topics, outfolder, min_f_score, feat_num, trans)
    if len(zero_topics) == num_topics:
        raise SystemExit("All topics are empty.")
    bgcl_dict = {bgc: sum(1 for g in genes if not g == '-')
                 for bgc, genes in bgc_dict.items()}
    bgc2topic = link_bgc_topics(lda, dict_lda, corpus_bow, bgcs, outfolder,
                                bgcl_dict, feat_scores, amplif=amplif)
    link_genes2topic(lda, dict_lda, outfolder)
    t_matches = retrieve_topic_matches(bgc2topic, feat_scores)
    top_match_file = os.path.join(outfolder, 'matches_per_topic.txt')
    t_matches = write_topic_matches(t_matches, bgc_classes, top_match_file,
                                    plot=False)
    t_matches = filter_matches(
        t_matches, feat_scores, filt_features, min_t_match, min_feat_match,
        overlap_score_cutoff)
    top_match_file_filt = top_match_file.split('.txt')[0] + '_filtered.txt'
    write_topic_matches(t_matches, bgc_classes, top_match_file_filt,
                        plot=plot)
    bgc_with_topics = retrieve_match_per_bgc(
        t_matches, bgc_classes, known_subcl, outfolder, plot=plot)

    # make filtered scatterplot
    lengths = []
    topics_per_bgc = []  # count amount of topics per bgc
    for bgc, val in bgc_with_topics.items():
        len_topics = 0
        bgclen = bgcl_dict[bgc]
        for match in val:
            probs = list(zip(*match[2]))[1]
            probs = [1 if p < 1 else round(p) for p in probs]
            if len(probs) > 1 or probs[0] > 1:
                # only count matches longer than 1
                len_topics += 1
            lengths.append((bgclen, sum(probs)))
        topics_per_bgc.append(len_topics)

    if lengths:
        # plot only when there are actual motif matches
        len_name = os.path.join(outfolder,
                                'len_bgcs_vs_len_topic_match_filtered.pdf')
        plot_topic_matches_lengths(lengths, len_name)
        # count amount of topics per bgc - filtered
        tpb_name = os.path.join(outfolder, 'topics_per_bgc_filtered.pdf')
        # add all the BGCs that do not have a match
        bgc_with_matches = set(bgc_with_topics.keys())
        topics_per_bgc += [0 for bgc in set(bgcs) if bgc not in
                           bgc_with_matches]
        topics_per_bgc_counts = Counter(topics_per_bgc)
        plot_topics_per_bgc(topics_per_bgc_counts, tpb_name)

        if plot:
            bgc_topic_heatmap(bgc_with_topics, bgc_classes, topic_num,
                              outfolder, metric='euclidean')
            bgc_topic_heatmap(bgc_with_topics, bgc_classes, topic_num,
                              outfolder, metric='correlation')
            bgc_class_heatmap(bgc_with_topics, bgc_classes, topic_num,
                              outfolder, metric='correlation')
    else:
        logger.info("Plots about stats could not be made as there were no matches")


def select_number_of_features(lda_topics, outfolder, min_f_score, feat_num,
                              trans):
    """Find list of features to use for each topic and write to topics.txt

    lda_topics: list of tuples, [(topic_number,features_string)]
    outfolder: str, path
    min_f_score: float, features will be selected until their cumulative
        score reaches this number
    feat_num: int, maximum amount of features to use
    trans: dict linking lda topics to topic names in ldavis if present
    filt_features: dict of set, for each topic the domains to use
        {topic: set(feats)}
    feat_scores: dict of dict: for each topic all features linked to their
        scores {topic: {feat:score} }
    zero_topics: list, storing topics that are empty
    """
    out_topics = os.path.join(outfolder, 'topics.txt')
    # to record the features as {topic:[(gene,prob)]}, features are selected
    # until the min_f_score or to feat_num as a maximum
    filt_features = {}
    feat_scores = {}
    zero_topics = []
    with open(out_topics, 'w') as outf:
        outf.write('Topic\tNumber_LDAvis\tTopic_length\tSelected_domains\t' +
                   'Domain_combinations\tScores\n')
        for top, mod in lda_topics:
            feat_scores[top] = {}
            nums = []
            doms = []
            for m in mod.split(' + '):
                num, dom = m.split('*')
                dom = dom.strip('"')
                num = float(num)
                if num == 0:
                    if not nums:
                        zero_topics.append(top)
                    break
                nums.append(num)
                doms.append(dom)
                feat_scores[top][dom] = num
            s = []
            m_len = len([s.append(num) for num in nums
                         if sum(s) < min_f_score])
            if m_len > feat_num:
                sel = feat_num
            else:
                sel = m_len
            filt_features[top] = set(doms[:sel])
            # write outfile
            sel_feats = zip(doms[:sel], nums[:sel])
            select_features = ','.join(a + ':' + str(b) for a, b in sel_feats)
            outf.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(top, trans[top],
                                                         sel, select_features,
                                                         ','.join(doms),
                                                         ','.join(
                                                             map(str, nums))))
    logger.info('{} empty topics'.format(len(zero_topics)))
    return filt_features, feat_scores, zero_topics


def link_bgc_topics(lda, dict_lda, corpus_bow, bgcs, outfolder, bgcl_dict,
                    feat_scores, amplif=False):
    """Returns dict of {bgc:{topic_num:[prob,[(gene,prob)],overlap_score]}}


    Writes file to outfolder/bgc_topics.txt and supplies plots if plot=True
    """
    logger.info('Linking topics to BGCs')
    doc_topics = os.path.join(outfolder, 'bgc_topics.txt')
    bgc2topic = {}
    if amplif:
        get_index = set(range(0, len(bgcs), amplif))
        bgc_bows = ((bgcs[i], corpus_bow[i]) for i in get_index)
    else:
        bgc_bows = zip(bgcs, corpus_bow)
    with open(doc_topics, 'w') as outf:
        for bgc, bgc_bow in bgc_bows:
            doc_doms = lda[bgc_bow]
            # doc_doms consists of three lists:
            # 1 topics 2 word2topic 3 word2topic with probability
            topd = {tpc: [prob, []] for tpc, prob in doc_doms[0]}
            for domcomb in doc_doms[2]:
                # find matching words with probabilities
                name = dict_lda[domcomb[0]]
                toptup = domcomb[1]  # all topic assignments for a word
                for t in toptup:
                    try:
                        # each t is a tuple of (topic, probability)
                        topd[t[0]][1].append((name, t[1]))
                    except KeyError:
                        # if this happens the term has such a low probability
                        # for this topic that it doesnt occur in doc_doms[0]
                        pass
            outf.write('>{}\n'.format(bgc))
            outf.write('len={}\n'.format(bgcl_dict[bgc]))
            for top, info in sorted(topd.items(), key=lambda x: x[1][0],
                                    reverse=True):
                # sort matching genes - high to low feature score in topic
                topic_scores = feat_scores.get(top, {})
                overlap_score = 0
                for feat in info[1]:
                    overlap_score += topic_scores.get(feat[0], 0)
                    # get because feat might have very low prob, gets left out
                gene_order = {feat[0]: i for i, feat in enumerate(sorted(
                    topic_scores.items(), key=itemgetter(1), reverse=True))}
                s_genes = sorted(info[1], key=lambda x: gene_order.get(
                    x[0], len(gene_order)))
                topd[top][1] = s_genes
                topd[top].append(overlap_score)
                genes = ','.join(
                    ['{}:{:.2f}'.format(g, p) for g, p in s_genes])
                string = 'topic={}\n\tp={:.3f}\n\toverlap_score={:.3f}'.format(
                    top, info[0], overlap_score) + \
                    '\n\tlen={}\n\tgenes={}\n'.format(len(info[1]), genes)
                outf.write(string)
            bgc2topic[bgc] = topd
    # if plot:
    # extract length of each bgc vs len of topic in each bgc
    logger.info('  plotting length of matches vs length of bgcs')
    lengths = ((bgcl_dict[bgc], len(val[t][1])) for bgc, val in
               bgc2topic.items() for t in val)
    len_name = os.path.join(outfolder, 'len_bgcs_vs_len_topic_match.pdf')
    plot_topic_matches_lengths(lengths, len_name)

    # count amount of topics per bgc
    tpb_name = os.path.join(outfolder, 'topics_per_bgc.pdf')
    topics_per_bgc = Counter([len(vals) for vals in bgc2topic.values()])
    plot_topics_per_bgc(topics_per_bgc, tpb_name)
    return bgc2topic


def plot_topic_matches_lengths(lengths, outname):
    """
    Make a scatterplot of the lengths of the topic matches vs the bgc lengths

    lengths: list of tuples, [(bgc_len,match_len)]
    outname: str, filepath
    """
    len_counts = Counter(lengths)
    x_y, counts = zip(*len_counts.items())
    bgc_len, topic_len = zip(*x_y)
    m_counts = max(len_counts.values())
    fig, ax = plt.subplots()
    scatter = ax.scatter(bgc_len, topic_len, c=sqrt(counts), s=2.5, vmin=1,
                         vmax=sqrt(m_counts), cmap='hot')
    if m_counts < 100:
        second_point = 1
    else:
        second_point = 20
    leg_range = [1] + [round(x, -1) for x in
                       range(second_point, m_counts + 1, ceil(m_counts / 4))]
    if len(leg_range) <= 4:
        leg_range.append(m_counts)
    leg_range = sorted(set(leg_range))
    kw = dict(num=leg_range, func=lambda c: c ** 2)
    legend = ax.legend(*scatter.legend_elements(**kw), loc='upper left',
                       title='Occurrence')
    ax.add_artist(legend)
    plt.xlabel('Length BGC')
    plt.ylabel('Length topic match')
    plt.title('Length of a BGC vs length of matching topic')
    plt.savefig(outname)
    plt.close()


def plot_topics_per_bgc(topics_per_bgc, outname):
    """Make a barplot of the amount of topics per bgc

    topics_per_bgc: dict/counter object, {n:bgcs_with_n_topics}
    outname: str
    """
    xs = range(max(topics_per_bgc) + 1)
    h = [topics_per_bgc[x] if x in topics_per_bgc else 0 for x in xs]
    plt.close()
    plt.bar(xs, h)
    plt.xlabel('Number of topics per BGC')
    plt.ylabel('Occurrence')
    plt.title('Topics per BGC')
    plt.savefig(outname)
    plt.close()


def link_genes2topic(lda, dict_lda, outfolder):
    """Write output file containing the genes (words/ dom-combis) per topic
    """
    outfile = os.path.join(outfolder, 'terms_to_topic.txt')
    with open(outfile, 'w') as outf:
        for d_id in dict_lda:
            d_name = dict_lda[d_id]
            domc_topics = sorted(lda.get_term_topics(d_name, 0.001),
                                 key=lambda x: x[1], reverse=True)
            dom_top_str = '\t'.join(';'.join(map(str, d)) for d in domc_topics)
            outf.write('{}\t{}\n'.format(d_name, dom_top_str))
        # todo? visualise amount of topics per term


def retrieve_topic_matches(bgc2topic, feat_scores):
    """Turns bgcs with matching topics to topics with matches from bgc

    bgc2topic: dict of {bgc:{'len':bgc_len,topic_num:[prob,[(gene,prob)],
        overlap_score]}}
    feat_scores: {topic:{genes:scores} }, dict of features /w scores for
        each topic
    topic_matches: {topic:[[prob,[(gene,prob)],bgc]]}
    """
    # get all topic matches per topic
    topic_matches = defaultdict(list)
    for bgc, dc in bgc2topic.items():
        for k, v in dc.items():
            if not k == 'len':
                ov_score = v.pop(-1)
                newv = v + [bgc, ov_score]
                topic_matches[k].append(newv)
    return topic_matches


def retrieve_match_per_bgc(topic_matches, bgc_classes, known_subcl, outfolder,
                           plot=True, cutoff=0.4):
    """
    Turns topics with matches back into bgc with matches and writes to file

    topic_matches: {topic:[[prob,(gene,prob),bgc,overlap_score]]}
    bgc_classes: {bgc:[class1,class2]}
    known_subcl: {bgc: [[info,domains]]}
    bgc2topic: dict of {bgc:[[topic_num,prob,[(gene,prob)]]]}

    Also compares for each match if it overlaps with a known subcluster
    """
    known_subcl_matches = defaultdict(list)
    bgc2topic = defaultdict(list)
    for topic, info in topic_matches.items():
        for match in info:
            bgc2topic[match[2]].append([topic] + match[:2] + [match[3]])
    with open(os.path.join(outfolder, 'bgc_topics_filtered.txt'), 'w') as outf:
        for bgc, info in sorted(bgc2topic.items()):
            bgc_class = bgc_classes.get(bgc, ['None'])[0]
            outf.write('>{}\nclass={}\n'.format(bgc, bgc_class))
            if known_subcl:
                if bgc in known_subcl:
                    # annotate if there are known subclusters in a bgc
                    for i, subcl in enumerate(known_subcl[bgc]):
                        outf.write('known_subcluster={}\n'.format(', '.join(
                            subcl)))
                # see if matches occur in a known subcluster
                matches_known = compare_known_subclusters(
                    known_subcl, bgc, bgc_class, info, cutoff=cutoff)
                for m_known in matches_known:
                    known_subcl_matches[m_known[0]].append(m_known[1:])
            for match in sorted(info, key=lambda x: x[1], reverse=True):
                outf.write('{}\t{:.3f}\t{:.3f}\t{}\n'.format(
                    match[0], match[1], match[3],
                    ','.join(['{}:{:.2f}'.format(m[0], m[1]) for m in match[2]
                              ])))
    if known_subcl:
        subcl_out = os.path.join(outfolder, 'known_subcluster_matches.txt')
        with open(subcl_out, 'w') as outf:
            # sort the subclusters alphabetically on first info element
            outf.write('##Values below each subcluster: %overlap len_overlap' +
                       ' bgc class topic topic_probability overlap_score' +
                       ' overlap_genes non_overlap_genes\n')
            for bgc, info in sorted(known_subcl.items(),
                                    key=lambda x: x[1][0][0]):
                for k_subclust in info:
                    outf.write('#{}\t{}\n'.format(
                        bgc, '\t'.join(map(str, k_subclust))))
                    overlap_list = known_subcl_matches[k_subclust[0]]
                    # give summary per topic?
                    # e.g. #topic x: 12 avg_overlap: 0.403
                    # sort from high to low overlap,topic,bgc
                    for m_overlap in sorted(overlap_list,
                                            key=lambda x: (-x[0], x[4], x[2])):
                        # overlap bgc class topic prob genes:prob
                        outf.write('{}\n'.format('\t'.join(
                            map(str, m_overlap))))
        # plot the overlap with known subcluster matches
        outname = os.path.join(outfolder,
                               'known_subcluster_matches_vs_cutoff.pdf')
        line_plot_known_matches(known_subcl_matches, outname,
                                cutoff=cutoff)
    return bgc2topic


def line_plot_known_matches(known_subcl_matches, outname, cutoff, steps=0.1):
    """Plot a line of the amount of known_subcl matches with different cutoffs


    Matches are only reported if at least two genes match, these can be two
    of the same genes if the prob is 1.5 or higher (close enough to two)
    """
    ys = [round(cutoff + i * steps, 1) for i in
          range(round((1.0 - cutoff) / steps) + 1)]
    xs = [0] * len(ys)
    for info in known_subcl_matches.values():
        if len(info) > 0:
            for i, thresh in enumerate(ys):
                for overlap in info:
                    if overlap[0] >= thresh and overlap[1] > 1:
                        xs[i] += 1
                        break
    logger.info(('\nThis method detects {} known sub-clusters with an overlap' +
           ' cutoff of {}. With all different overlap cutoffs:').format(
        xs[2], ys[2]))
    logger.info(', '.join(map(str, ys)))
    logger.info(', '.join(map(str, xs)))
    fig, ax = plt.subplots()
    line = ax.plot(ys, xs)
    ax.set_ylim(0, len(known_subcl_matches))
    plt.xlabel('Overlap threshold')
    plt.ylabel('Characterised subclusters with a match')
    plt.title(
        'Number of characterised subclusters with a match according\n\
        to different overlap thresholds')
    plt.savefig(outname)
    plt.close()


def compare_known_subclusters(known_subcl, bgc, bgc_class, matches, cutoff):
    """Find % overlap with known subclusters and returns it as a list

    known_subcl: {bgc: [[info,domains]]
    bgc: str, bgcname
    bgc_class: str, class of bgc
    matches: [[topic_num,prob,[(gene,prob)],overlap_score]]
    cutoff: float, overlap cutoff used for reporting
    matches_overlap: [[first_info_element,%overlap,len_overlap,bgc,bgc_class,
        topic_num,prob,overlapping_genes,non_overlapping_genes]]
    """
    matches_overlap = []
    for match in matches:
        g_list = match[2]
        doms = set(list(zip(*g_list))[0])
        for k_subs in known_subcl.values():
            for k_sub in k_subs:
                k_list = k_sub[-1].split(',')
                k_sub_doms = set(k_sub[-1].split(','))
                if '-' in k_sub_doms:
                    k_sub_doms.remove('-')
                    k_list = [k for k in k_list if not k == '-']
                overl_d_set = doms & k_sub_doms
                l_overlap = len(overl_d_set)
                if not len(k_sub_doms) - len(k_list) == 0:
                    # there are doms in the k-subcl that are duplicated
                    dupls = [kc for kc in Counter(k_list).items() if kc[1] > 1]
                    add_overl = 0
                    for dom, count in dupls:
                        if dom in doms:
                            overl_domtups = [domt for domt in g_list
                                             if domt[0] == dom]
                            for overl_domtup in overl_domtups:
                                if round(overl_domtup[1]) >= count:
                                    l_overlap += count - 1
                overlap = l_overlap / len(k_list)
                if overlap > cutoff and len(k_list) > 1:
                    match_overl_genes = [(g, p,) for g, p in
                                         g_list if g in overl_d_set]
                    overl_d = ','.join(sorted(['{}:{:.2f}'.format(g, p) for
                                               g, p in match_overl_genes]))
                    non_overl_d = ','.join(sorted(['{}:{:.2f}'.format(g, p)
                                                   for g, p in g_list if
                                                   g not in overl_d_set]))

                    matches_overlap.append([k_sub[0], round(overlap, 3),
                                            l_overlap, bgc, bgc_class,
                                            match[0], round(match[1], 3),
                                            round(match[3], 3), overl_d,
                                            non_overl_d])
    return matches_overlap


def write_topic_matches(topic_matches, bgc_classes, outname, plot):
    """Writes topic matches to a file sorted on length and alphabet

    topic_matches: {topic:[[prob,[(gene,prob)],bgc,overlap_score]]}
    bgc_classes: {bgc: [class1,class2]}
    outname: str, filepath
    """
    logger.info('Writing matches to {}'.format(outname))
    # a set of bgc classes
    s_b_c = set([v for vals in bgc_classes.values() for v in vals])
    s_b_c.add('None')
    plotlines = pd.DataFrame(columns=sorted(s_b_c))
    plotlines_1 = pd.DataFrame(columns=sorted(s_b_c))
    # occurence of each topic
    prevl = {t: len(vals) for t, vals in topic_matches.items()}
    sumfile = outname.split('.txt')[0] + '_summary.txt'
    with open(outname, 'w') as outf, open(sumfile, 'w') as sumf:
        sumf.write('Topic\tmatches\tmatches_len>1\tclasses\tclasses_len>1\n')
        for topic, matches in sorted(topic_matches.items()):
            classes = Counter()  # classes for all matches
            classes_1 = Counter()  # classes for matches longer than 1
            for p, g, bgc, overlap in matches:
                bgc_class = bgc_classes.get(bgc, ['None'])[0]
                classes.update([bgc_class])
                try:
                    if len(g) > 1 or round(g[0][1]) > 1:
                        classes_1.update([bgc_class])
                except IndexError:
                    # there is a probability for a match to the topic but
                    # it is so low there are no genes in the match
                    pass
            for count_class, count in classes.items():
                plotlines.loc[topic, count_class] = count
                plotlines_1.loc[topic, count_class] = classes_1[count_class]
            # sort classes
            class_str = ','.join([':'.join(map(str, cls)) for cls in
                                  sorted(classes.items(),
                                         key=lambda x: (-x[1], x[0]))])
            class1_str = ','.join([':'.join(map(str, cls)) for cls in
                                   sorted(classes_1.items(),
                                          key=lambda x: (-x[1], x[0]))])
            prevl = len(matches)
            prevl_bigger_1 = sum(classes_1.values())
            # topicnr matches matches>1 classes classes>1
            outf.write('#Topic {}, matches:{}, matches_len>1:{}'.format(
                topic, prevl, prevl_bigger_1) +
                ', classes:{}, classes_len>1:{}\n'.format(
                    class_str, class1_str))
            sum_line = [topic, prevl, prevl_bigger_1, class_str, class1_str]
            sumf.write('{}\n'.format('\t'.join(map(str, sum_line))))
            # sort the matches by length and then by alphabet
            try:
                sorted_matches = sorted(
                    matches, key=lambda x:(len(x[1]), list(zip(*x[1]))[0]))
            except IndexError:
                pass  # there is no match same as above
            else:
                topic_matches[topic] = sorted_matches
                for match in sorted_matches:
                    outf.write('{:.3f}\t{:.3f}\t{}\t{}\t{}\n'.format(
                        match[0], match[3], ','.join(
                            ['{}:{:.2f}'.format(m[0], m[1]) for m in match[1]]
                            ), match[2],
                        bgc_classes.get(match[2], ['None'])[0]))
    if plot:
        bplot_name = os.path.join(os.path.split(outname)[0], 'topic_stats.pdf')
        barplot_topic_stats(plotlines, bplot_name)
        bplot_name_1 = os.path.join(os.path.split(outname)[0],
                                    'topic_stats_matches>1.pdf')
        barplot_topic_stats(plotlines_1, bplot_name_1)
    return topic_matches


def barplot_topic_stats(df, outname):
    """makes a stacked barplot of the classes in df for each topic

    df: pandas dataframe with    index as topic numbers and columns as classes
    outname: str, filepath
    """
    logger.info('  making barplot of topic stats')
    df = df.fillna(0).infer_objects(copy=False)
    len_no_none = len(df.columns) - 1
    if len(df.columns) > 10:
        cols = sns.cubehelix_palette(len_no_none, start=1.2, rot=2,
                                     dark=0.11, light=0.85)
    else:
        cols = sns.color_palette()[:len_no_none]
    # make None always white
    non_i = [i for i, non in enumerate(df.columns) if non == 'None'][0]
    colours = cols[:non_i] + ['w'] + cols[non_i:]
    ax = df.plot.bar(stacked=True, color=colours, edgecolor='#333333',
                     width=1.0)
    legend = ax.legend(loc='best', fontsize='x-small', title='BGC class')
    ax.add_artist(legend)
    ax.tick_params(axis='x', which='major', labelsize=4)
    ax.tick_params(axis='x', which='minor', labelsize=4)
    plt.xlabel('Topics')
    plt.ylabel('Occurence')
    plt.title('BGC class distribution across topics')
    plt.savefig(outname)
    plt.close()


def filter_matches(topic_matches, feat_scores, filt_features, min_t_match,
                   min_feat_match, overlap_score_cutoff=0.15,
                   total_feat_score_cutoff=1.1):
    """Filters topic_matches based on cutoffs

    topic_matches: {topic:[[prob,(gene,prob)],bgc,overlap_score]}, topic
        linked to matches
    feat_scores: {topic:{genes:scores} }, dict of features /w scores for
        each topic
    filt_features: {topic:set(genes)}, dict of sets of feats to use for each
        topic
    min_t_match: float, minimal score of a topic matching a bgc
    min_feat_match: float, minimal score of a feature matching in a topic in
        a bgc
    overlap_score_cutoff: float, minimal fraction of the topic that should be
        present in the topic (subcluster) match
    total_feat_score_cutoff: float, minimal amount of features (based on sum
        of their feature scores) that should be present in subcluster
    filt_topic_matches: {topic:[[prob,(gene,prob)],bgc,overlap_score]}
    """
    logger.info('Filtering matches')
    filt_topic_matches = defaultdict(list)
    for topic, matches in topic_matches.items():
        # filt_topic_matches[topic] = []
        try:
            feats_dict = feat_scores[topic]
            use_feats = filt_features[topic]
        except KeyError:
            # topic is empty
            feats_dict = {}
            use_feats = {}
        else:
            use_feats = set(feats_dict.keys())
        for match in matches:
            match_p = match[0]
            newfeats = []
            overlap_score = 0
            total_feat_scores = 0  # proxy for amount of genes in the subcl
            for feat in match[1]:
                dom_com = feat[0]
                if dom_com in use_feats and feat[1] >= min_feat_match:
                    newfeats.append(feat)
                    total_feat_scores += feat[1]
                    overlap_score += feats_dict[dom_com]
            # instead of filtering on probability (min_t_match), we filter on
            # sum of the newfeats scores as bgcs can be very large
            # removed: match_p > min_t_match
            # if len(newfeats) > 2 and overlap_score > overlap_score_cutoff:
            if total_feat_scores > total_feat_score_cutoff and \
                    overlap_score > overlap_score_cutoff:
                if newfeats:
                    bgc = match[2]
                    filt_topic_matches[topic].append([match_p, newfeats, bgc,
                                                      overlap_score])
    return filt_topic_matches


def bgc_topic_heatmap(bgc_with_topic, bgc_classes, topic_num, outfolder,
                      metric='euclidean'):
    """Make a clustered heatmap of bgcs and topics, and optional bgc_classes

    bgc_with_topic: dict of {bgc:[[topic_num,prob,[(gene,prob)]]]}
    bgc_classes: dict of {bgc:[[class1,class2]]}
    topic_num: int, number of topics in the model

    """
    logger.info('Making clustered heatmap, metric: {}'.format(metric))
    # make pd dataframe from bgc with topic with prob as value for present tpic
    bgcs, topics = zip(*bgc_with_topic.items())
    data = [{v[0]: v[1] for v in val} for val in topics]
    df = pd.DataFrame(data, index=bgcs, columns=list(range(topic_num)))
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(0).infer_objects(copy=False)
    # colour rows by bgc class
    class_set = set(bgc_classes.keys())
    labels = [bgc_classes[bgc][0] if bgc in class_set else 'None' for bgc
              in bgcs]
    s_labels = sorted(set(labels))
    # get colours
    if 'None' in s_labels:
        s_labels.remove("None")
    if len(s_labels) > 10:
        lut = dict(zip(s_labels, sns.cubehelix_palette(len(
            s_labels), start=1.2, rot=2, dark=0.11, light=0.85)))
    else:
        lut = dict(zip(s_labels, sns.color_palette()))
    lut['None'] = 'w'  # make None always white
    s_labels = ['None'] + s_labels
    row_labs = pd.DataFrame(labels, index=bgcs, columns=['BGC classes'])
    row_colours = row_labs['BGC classes'].map(lut)  # map colour to a label

    assert not df.isnull().values.any() # DEBUG
    assert np.isfinite(df.values).all() # DEBUG

    g = sns.clustermap(df, cmap='nipy_spectral', row_colors=row_colours,
                       linewidths=0, metric=metric, yticklabels=False,
                       xticklabels=True,
                       cbar_kws={'orientation': 'horizontal'}, vmin=0, vmax=1)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(),
                                 fontsize=5)
    # don't show dendrograms
    g.ax_col_dendrogram.set_visible(False)
    g.ax_row_dendrogram.set_ylim([0, 0.00001])
    g.ax_row_dendrogram.set_xlim([0, 0.00001])
    # make legend for classes
    for label in s_labels:
        g.ax_row_dendrogram.bar(0, 0, color=lut[label], label=label,
                                linewidth=0)
    g.ax_row_dendrogram.legend(loc="center left", fontsize='small',
                               title='BGC classes')
    # move colourbar
    g.cax.set_position([.35, .78, .45, .0225])
    plt.savefig(
        os.path.join(outfolder, 'topic_heatmap_{}.pdf'.format(metric)))
    plt.close()


def bgc_class_heatmap(bgc_with_topic, bgc_classes, topic_num, outfolder,
                      metric='correlation'):
    """Make a clustered heatmap of bgcs and topics, and optional bgc_classes

    bgc_with_topic: dict of {bgc:[[topic_num,prob,[(gene,prob)]]]}
    bgc_classes: dict of {bgc:[[class1,class2]]}
    topic_num: int, number of topics in the model

    """
    logger.info('Making clustered heatmap of classes, metric: {}'.format(metric))
    # make pd dataframe from bgc with topic with prob as value for present tpic
    bgcs, topics = zip(*bgc_with_topic.items())
    data = [{v[0]: v[1] for v in val} for val in topics]
    df = pd.DataFrame(data, index=bgcs, columns=list(range(topic_num)))
    df = df.fillna(0).infer_objects(copy=False)
    # colour rows by bgc class
    class_set = set(bgc_classes.keys())
    labels = [bgc_classes[bgc][0] if bgc in class_set else 'None' for bgc
              in bgcs]
    s_labels = sorted(set(labels))
    # cluster each class (hierarchical, correlation)
    class_i = clust_class_bgcs(df, labels, s_labels)
    # get colours
    if 'None' in s_labels:
        s_labels.remove("None")
    if len(s_labels) > 10:
        lut = dict(zip(s_labels, sns.cubehelix_palette(len(
            s_labels), start=1.2, rot=2, dark=0.11, light=0.85)))
    else:
        lut = dict(zip(s_labels, sns.color_palette()))
    lut['None'] = 'w'  # make None always white
    s_labels = ['None'] + s_labels
    row_labs = pd.DataFrame(labels, index=bgcs, columns=['BGC classes'])
    row_colours = row_labs.iloc[class_i, 0].map(lut)  # map colour to a label

    assert not df.isnull().values.any() # DEBUG
    assert np.isfinite(df.values).all() # DEBUG

    g = sns.clustermap(df.iloc[class_i, :], cmap='nipy_spectral',
                       row_colors=row_colours, linewidths=0, metric=metric,
                       yticklabels=False, xticklabels=True,
                       cbar_kws={'orientation': 'horizontal'}, vmin=0, vmax=1,
                       row_cluster=False)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(),
                                 fontsize=5)
    # don't show dendrograms
    g.ax_col_dendrogram.set_visible(False)
    g.ax_row_dendrogram.set_ylim([0, 0.00001])
    g.ax_row_dendrogram.set_xlim([0, 0.00001])
    # make legend for classes
    for label in s_labels:
        g.ax_row_dendrogram.bar(0, 0, color=lut[label], label=label,
                                linewidth=0)
    g.ax_row_dendrogram.legend(loc="center left", fontsize='small',
                               title='BGC classes')
    # move colourbar
    g.cax.set_position([.35, .78, .45, .0225])
    plt.savefig(
        os.path.join(outfolder, 'class-topic_heatmap_{}.pdf'.format(metric)))
    plt.close()


def clust_class_bgcs(df, labels, s_labels):
    """Returns a list of indices ordered on clustered classes
    """
    # get a list of clustered indexes for all and then add them
    inds = np.array([], dtype='int32')
    for bgc_class in s_labels:
        c_i = [i for i, cls in enumerate(labels) if cls == bgc_class]
        dist = sch.distance.pdist(df.iloc[c_i, :], metric='correlation')
        clust = sch.linkage(dist, metric='correlation')
        ind = sch.leaves_list(clust)
        # print(ind)
        ind_reorder = [c_i[i] for i in ind]
        inds = np.append(inds, ind_reorder)
    return inds


def read2dict(filepath, sep=',', header=False):
    """Read file into a dict {first_column:[other_columns]}

    filepath: str
    sep: str, delimiter in the file
    header: bool, ignore first line
    """
    output = {}
    with open(filepath, 'r') as inf:
        if header:
            inf.readline()
        for line in inf:
            line = line.strip().split(sep)
            output[line[0]] = line[1:]
    return output


def plot_convergence(logfile, iterations, outfile):
    """
    Plot convergence of log_likelihood of the model as calculated in logging

    logfile: str, filepath
    iterations: int
    """
    p = re.compile(r"(-*\d+\.\d+) per-word .* (\d+\.\d+) perplexity")
    matches = [p.findall(ln) for ln in open(logfile)]
    matches = [m for m in matches if len(m) > 0]
    tuples = [t[0] for t in matches]
    if tuples:
        perplexity = [float(t[1]) for t in tuples]
        liklihood = [float(t[0]) for t in tuples]
        eval_evry = iterations / len(tuples)
        iters = [eval_evry * i for i in range(len(tuples))]
        plt.plot(iters, liklihood, c="black")
        plt.ylabel("log likelihood")
        plt.xlabel("iteration")
        plt.title("Topic Model Convergence")
        plt.grid()
        plt.savefig(outfile)
        # plt.show
        plt.close()
    else:
        logger.error("convergence of model could not be plot; likelihood not in"
              " log file")

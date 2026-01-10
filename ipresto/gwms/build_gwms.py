import logging

logger = logging.getLogger(__name__)

def build_motif_gwms(
    label2geneprobs,
    mm,
    mgc,
    ct,
    mgp,
    out_dirpath,
    ):
    logger.info(f"Building GWMs for mm={mm}, mgc={mgc}, ct={ct}, mgp={mgp}...")



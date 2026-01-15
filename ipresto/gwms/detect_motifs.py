import logging
from collections import defaultdict
from dataclasses import dataclass
from ipresto.gwms.subcluster_motif import SubclusterMotif

logger = logging.getLogger(__name__)

@dataclass
class MotifHit:
    bgc_id: str
    motif_id: str
    score: float
    tokenized_genes: set

def parse_clusters_file(clusters_file):
    with open(clusters_file, "r") as infile:
        clusters = [read_cluster(line) for line in infile]
        clusters = {bgc_id: bgc_genes for bgc_id, bgc_genes in clusters}
    return clusters


def read_cluster(line):
    bgc_id, tokenized_genes = line.rstrip().split(",", 1)
    tokenized_genes = set(tokenized_genes.split(","))
    tokenized_genes.discard("-")  # genes without biosynthetic domains
    return bgc_id, tokenized_genes


def parse_motifs_file(motifs_file):
    subcluster_motifs = dict()
    infile = open(motifs_file, "r")
    while True:
        # read 4 lines at a time
        lines = [infile.readline().rstrip() for _ in range(4)]
        # stop it end of file
        if not lines[0]:
            break
        # add subcluster motif
        motif = SubclusterMotif.from_lines(lines)
        subcluster_motifs[motif.motif_id] = motif
    infile.close()
    return subcluster_motifs


def write_motif_hits(motif_hits, output_filepath):

    with open(output_filepath, "w") as outfile:
        # print header
        header_fields = [
            "bgc_id",
            "motif_id",
            "n_training_matches",
            "score_threshold",
            "score",
            "hit_genes",
        ]
        print("\t".join(header_fields), file=outfile)
        # print lines
        for bgc_id, hits in motif_hits.items():
            for motif_hit in hits:
                n_training_matches = motifs[motif_hit.motif_id].n_matches
                threshold = motifs[motif_hit.motif_id].threshold
                line_fields = [
                    motif_hit.bgc_id,
                    motif_hit.motif_id,
                    str(n_training_matches),
                    str(threshold),
                    str(motif_hit.score),
                    ",".join(sorted(motif_hit.tokenized_genes)),
                ]
            print("\t".join(line_fields), file=outfile)


def detect_motifs(clusters_filepath, motifs_filepath, output_filepath):

    clusters = parse_clusters_file(clusters_filepath)
    motifs = parse_motifs_file(motifs_filepath)
    logger.info(f"Loaded {len(clusters)} clusters from {clusters_filepath} and {len(motifs)} subcluster motifs from {motifs_filepath}   ")

    motif_hits = defaultdict(list)
    for bgc_id, bgc_genes in clusters.items():
        for motif in motifs.values():
            score = motif.calculate_score(bgc_genes)
            if score < motif.threshold:
                continue

            hit_genes = set(motif.tokenized_genes) & bgc_genes
            if len(hit_genes) < 2:
                continue

            motif_hits[bgc_id].append(
                MotifHit(bgc_id, motif.motif_id, score, hit_genes)
                )

    logger.info(
        f"Wrote {sum(len(v) for v in motif_hits.values())} motif hits meeting "
        f"threshold criteria to {output_filepath}")

    return motif_hits
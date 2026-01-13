import logging
from dataclasses import dataclass
from ipresto.gwms.subcluster_motif import SubclusterMotif

logger = logging.getLogger(__name__)

@dataclass
class MotifHit:
    bgc_id: str
    motif_id: str
    score: float
    genes: set

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


def detect_motifs(clusters_filepath, motifs_filepath, output_filepath):

    logger.info(f"Starting motif detection: scanning {clusters_filepath} against subcluster motifs in {motifs_filepath}")

    clusters = parse_clusters_file(clusters_filepath)
    motifs = parse_motifs_file(motifs_filepath)

    logger.info(f"Loaded {len(clusters)} biosynthetic gene clusters and {len(motifs)} subcluster motifs")

    motif_hits = []
    for bgc_id, bgc_genes in clusters.items():
        for motif in motifs.values():
            score = motif.calculate_score(bgc_genes)
            if score < motif.threshold:
                continue

            common_genes = set(motif.tokenized_genes) & bgc_genes
            if len(common_genes) < 2:
                continue

            motif_hits.append(
                MotifHit(bgc_id, motif.motif_id, score, common_genes)
                )
    logger.info(f"Identified {len(motif_hits)} motif hits meeting threshold criteria")


    with open(output_filepath, "w") as outfile:
        # print header
        header_fields = [
            "bgc_id",
            "motif_id",
            "n_training",
            "score_threshold",
            "score",
            "genes",
        ]
        print("\t".join(header_fields), file=outfile)
        # print lines
        for motif_hit in motif_hits:
            n_training = motifs[motif_hit.motif_id].n_matches
            threshold = motifs[motif_hit.motif_id].threshold
            line_fields = [
                motif_hit.bgc_id,
                motif_hit.motif_id,
                str(n_training),
                str(threshold),
                str(motif_hit.score),
                ",".join(sorted(motif_hit.genes)),
            ]
            print("\t".join(line_fields), file=outfile)

    logger.info(f"Results written to {output_filepath}")

    return motif_hits
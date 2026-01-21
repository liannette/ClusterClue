from collections import Counter
from clusterclue.preprocess.utils import (
    count_non_emtpy_genes,
    parse_cluster_line,
    format_cluster_to_string,
    write_gene_counts,
    write_representatives,
    read_txt,
)


def test_count_non_empty_genes():
    genes = [("A", "B"), ("-",), ("C",)]
    assert count_non_emtpy_genes(genes) == 2

    genes = [("-",), ("-",)]
    assert count_non_emtpy_genes(genes) == 0

    genes = []
    assert count_non_emtpy_genes(genes) == 0


def test_parse_cluster_line():
    line = "cluster1,domainA;domainB,domainC;domainD"
    cluster_id, genes = parse_cluster_line(line)
    assert cluster_id == "cluster1"
    assert genes == [("domainA", "domainB"), ("domainC", "domainD")]

    line = "cluster2,-"
    cluster_id, genes = parse_cluster_line(line)
    assert cluster_id == "cluster2"
    assert genes == [("-",)]


def test_format_cluster_to_string():
    cluster_id = "cluster1"
    genes = [("A", "B"), ("C", "D")]
    formatted = format_cluster_to_string(cluster_id, genes)
    assert formatted == "cluster1,A;B,C;D\n"

    cluster_id = "cluster2"
    genes = [("-",)]
    formatted = format_cluster_to_string(cluster_id, genes)
    assert formatted == "cluster2,-\n"


def test_write_gene_counts(tmp_path):
    gene_counter = Counter({("domainB",): 2, ("domainA", "domainB"): 3})
    outfile_path = tmp_path / "gene_counts.txt"
    write_gene_counts(gene_counter, outfile_path)

    with open(outfile_path, "r") as f:
        lines = f.readlines()

    assert lines[0] == "#Total\t5\n"
    assert lines[1] == "domainA;domainB\t3\n"
    assert lines[2] == "domainB\t2\n"


def test_write_representatives(tmp_path):
    representative_clusters = {
        "cluster1": ["cluster1", "cluster2"],
        "cluster3": ["cluster3", "cluster4"],
    }
    outfile_path = tmp_path / "representatives.txt"
    write_representatives(representative_clusters, outfile_path)

    with open(outfile_path, "r") as f:
        lines = f.readlines()

    assert lines[0] == "cluster1\tcluster1,cluster2\n"
    assert lines[1] == "cluster3\tcluster3,cluster4\n"


def test_read_txt(tmp_path):
    file_path = tmp_path / "test.txt"
    lines_to_write = ["line1", "line2", "line3"]
    with open(file_path, "w") as f:
        f.write("\n".join(lines_to_write))

    lines_read = read_txt(file_path)
    assert lines_read == lines_to_write

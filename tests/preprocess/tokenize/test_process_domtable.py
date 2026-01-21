from pathlib import Path

from clusterclue.preprocess.tokenize.process_domtable import (
    calculate_overlap,
    domains_are_overlapping,
    domtable_to_tokenized_cluster,
    process_domtables,
)


def test_calculate_overlap():
    assert calculate_overlap((0, 5), (4, 8)) == 1
    assert calculate_overlap((0, 5), (5, 8)) == 0
    assert calculate_overlap((0, 5), (6, 8)) == -1
    assert calculate_overlap((0, 5), (0, 5)) == 5
    assert calculate_overlap((0, 10), (5, 7)) == 2


def test_domains_are_overlapping():
    # small overlap
    assert domains_are_overlapping((0, 5), (4, 8), 0.1) is True
    assert domains_are_overlapping((0, 5), (4, 8), 0.5) is False
    # no overlap
    assert domains_are_overlapping((0, 5), (5, 8), 0) is False
    # complete overlap
    assert domains_are_overlapping((0, 5), (0, 5), 1) is False
    assert domains_are_overlapping((0, 10), (5, 7), 0.1) is True


def test_domtable_to_tokenised_cluster():
    test_domtable_file_path = (
        Path(__file__).parent / "test_data/input/BGC0002659.domtable"
    )

    # Test the output of the function
    cluster_id, tokenized_genes, domain_hits = domtable_to_tokenized_cluster(
        domtable_path=test_domtable_file_path,
        max_domain_overlap=0.1,
    )

    assert cluster_id == "BGC0002659"
    assert len(tokenized_genes) == 5
    assert tokenized_genes == [
        ("-",),
        ("AzlC",),
        ("2OG-Fe_Oxy_2",),
        ("Radical_SAM_c10", "SPASM_c22"),
        ("-",),
    ]
    assert len(domain_hits) == 4

    # Test the output of the function for an domtable file with no matches
    test_domtable_file_path = Path(__file__).parent / "test_data/input/empty.domtable"
    cluster_id, tokenized_genes, domain_hits = domtable_to_tokenized_cluster(
        domtable_path=test_domtable_file_path,
        max_domain_overlap=0.1,
    )
    assert cluster_id == "empty"
    assert len(tokenized_genes) == 0
    assert len(domain_hits) == 0


def test_process_domtables(tmp_path):
    domtables_dir_path = Path(__file__).parent / "test_data/input/domtables"
    clusters_path = tmp_path / "clusterfile.csv"
    gene_counts_path = tmp_path / "gene_counts.csv"
    domain_hits_path = tmp_path / "all_domain_hits.txt"
    min_genes = 2
    max_domain_overlap = 0.1
    cores = 1
    verbose = False

    process_domtables(
        domtables_dir_path=domtables_dir_path,
        cluster_file_path=clusters_path,
        gene_counts_file_path=gene_counts_path,
        domain_hits_file_path=domain_hits_path,
        min_genes=min_genes,
        max_domain_overlap=max_domain_overlap,
        cores=cores,
        verbose=verbose,
    )

    expected_clusters_path = (
        Path(__file__).parent / "test_data/expected_output/clusterfile.csv"
    )
    with open(clusters_path, "r") as out_file:
        with open(expected_clusters_path, "r") as expected_file:
            assert out_file.read() == expected_file.read()

    expected_counts_path = (
        Path(__file__).parent / "test_data/expected_output/gene_counts.csv"
    )
    with open(gene_counts_path, "r") as out_file:
        with open(expected_counts_path, "r") as expected_file:
            assert out_file.read() == expected_file.read()

    expected_domain_hits_path = (
        Path(__file__).parent / "test_data/expected_output/all_domain_hits.txt"
    )
    with open(domain_hits_path, "r") as out_file:
        with open(expected_domain_hits_path, "r") as expected_file:
            assert out_file.read() == expected_file.read()

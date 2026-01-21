from pathlib import Path

from clusterclue.preprocess.domain_filtering import (
    extract_domain_base_name,
    filter_domains_in_gene,
    perform_domain_filtering,
)


def test_extract_domain_base_name():
    assert extract_domain_base_name("domain_c1") == "domain"
    assert extract_domain_base_name("domain_c2") == "domain"
    assert extract_domain_base_name("domain") == "domain"
    assert extract_domain_base_name("domain_c") == "domain_c"


def test_filter_domains_in_gene():
    include_domains = {"domain1", "domain2"}
    gene = ("domain1", "domain2_c1", "domain3_c2")
    assert filter_domains_in_gene(gene, include_domains) == ("domain1", "domain2_c1")

    gene = ("domain4", "domain5_c1")
    assert filter_domains_in_gene(gene, include_domains) == ("-",)


def test_perform_domain_filtering(tmp_path):
    in_file_path = Path(__file__).parent / "test_data/input/clusterfile_unfiltered.csv"
    domain_filter_file_path = (
        Path(__file__).parent / "test_data/input/biosynthetic_domains.txt"
    )

    # test with min_genes = 2
    min_genes = 2
    cores = 1
    verbose = False
    filtered_clusters_path = tmp_path / "out_file.csv"
    gene_counts_path = tmp_path / "gene_counts.csv"
    perform_domain_filtering(
        in_file_path,
        domain_filter_file_path,
        filtered_clusters_path,
        gene_counts_path,
        min_genes,
        cores,
        verbose,
    )

    expected_filtered_clusters_path = (
        Path(__file__).parent / "test_data/expected_output/clusterfile_filtered.csv"
    )
    with open(filtered_clusters_path, "r") as out_file:
        with open(expected_filtered_clusters_path, "r") as expected_file:
            assert out_file.read() == expected_file.read()

    expected_counts_path = (
        Path(__file__).parent / "test_data/expected_output/gene_counts_filtered.csv"
    )
    with open(gene_counts_path, "r") as out_file:
        with open(expected_counts_path, "r") as expected_file:
            assert out_file.read() == expected_file.read()

import networkx as nx
import os

from clusterclue.preprocess.similarity_filtering import (
    generate_adjacent_domain_pairs,
    calc_adj_index,
    is_contained,
    extract_domains,
    generate_edges,
    generate_edge,
    generate_graph,
    read_edges,
    find_representatives,
    find_all_representatives,
    calculate_domain_lengths,
    get_representatives,
    filter_similar_clusters,
)


def test_generate_adjacent_domain_pairs():
    domains = ["domA", "domB", "domC", "-", "domE", "domD"]
    pairs = generate_adjacent_domain_pairs(domains)
    assert len(pairs) == 3
    assert ("domA", "domB") in pairs
    assert ("domB", "domC") in pairs
    assert ("domD", "domE") in pairs


def test_calc_adj_index():
    domains1 = ["domA", "domB"]
    domains2 = ["domA", "domB"]
    assert calc_adj_index(domains1, domains2) == 1.0

    domains1 = ["domA", "domB"]
    domains2 = ["domA", "domB", "domC"]
    assert calc_adj_index(domains1, domains2) == 0.5

    domains1 = ["domA", "domB", "domC"]
    domains2 = ["domA", "domB", "domD"]
    assert calc_adj_index(domains1, domains2) == 1 / 3

    domains1 = ["domA", "domB"]
    domains2 = ["domA", "-", "domB"]
    assert calc_adj_index(domains1, domains2) == 0.0


def test_is_contained():
    domains1 = ["domA", "domB", "domC"]
    domains2 = ["domA", "domB", "domC", "domD"]
    assert is_contained(domains1, domains2) is True
    assert is_contained(domains2, domains1) is True
    domains3 = ["domA", "domB", "domD"]
    assert is_contained(domains1, domains3) is False
    domains4 = ["domA", "domB", "-"]
    assert is_contained(domains4, domains1) is True


def test_extract_domais():
    tokenized_genes = [("domA",), ("domB", "domC"), ("-",)]
    domains = extract_domains(tokenized_genes)
    assert domains == ["domA", "domB", "domC", "-"]


def test_generate_edge():
    pair = (("bgc1", [("domA",), ("domB",)]), ("bgc2", [("domA",), ("domB", "domC")]))
    edge = generate_edge(pair, adj_cutoff=0.3)
    assert edge == ("bgc1", "bgc2", 0.5, True)

    pair = (
        ("bgc1", [("domA",), ("domB", "domC")]),
        ("bgc2", [("domA",), ("domB", "domD")]),
    )
    edge = generate_edge(pair, adj_cutoff=0.3)
    assert edge == ("bgc1", "bgc2", 1 / 3, False)


def test_generate_edges():
    dom_dict = {
        "bgc1": [("domA",), ("domB",)],
        "bgc2": [("domA",), ("domB", "domC")],
        "bgc3": [("domA",), ("domB", "domD")],
    }
    cutoff = 0.5
    cores = 1
    temp_file = "temp_edges.txt"
    verbose = False
    generate_edges(dom_dict, cutoff, cores, temp_file, verbose)
    edges = list(read_edges(temp_file))
    assert len(edges) == 2
    assert edges[0][0] == "bgc1"
    assert edges[0][1] == "bgc2"
    assert edges[1][0] == "bgc1"
    assert edges[1][1] == "bgc3"
    os.remove(temp_file)


def test_generate_graph():
    edges = [("bgc1", "bgc2", {"ai": 0.5, "contained": False})]
    graph = generate_graph(edges, verbose=False)
    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 1


def test_read_edges():
    file_path = "temp_edges.txt"
    with open(file_path, "w") as f:
        f.write("bgc1\tbgc2\t0.5\tFalse\n")
    edges = list(read_edges(file_path))
    assert len(edges) == 1
    assert edges[0][0] == "bgc1"
    assert edges[0][1] == "bgc2"
    os.remove(file_path)


def test_find_representatives():
    clqs = [
        ["bgc1", "bgc2"],
        ["bgc3",],
    ]
    domain_length_dict = {"bgc1": 1, "bgc2": 2, "bgc3": 3}
    graph = nx.Graph()
    graph.add_edges_from([("bgc1", "bgc2")])
    reps = find_representatives(clqs, domain_length_dict, graph)
    assert len(reps) == 2
    assert "bgc2" in reps
    assert "bgc3" in reps

    domain_length_dict = {"bgc1": 3, "bgc2": 3, "bgc3": 2}
    graph = nx.Graph()
    graph.add_edges_from([("bgc1", "bgc2")])
    reps = find_all_representatives(domain_length_dict, graph)
    assert len(reps) == 1
    assert "bgc1" in reps or "bgc2" in reps


def test_calculate_domain_lengths():
    tokenised_clusters = {
        "bgc1": [("domA",), ("domB", "domC")],
        "bgc2": [("domA",), ("-",)],
    }
    lengths = calculate_domain_lengths(tokenised_clusters)
    assert lengths == {"bgc1": 3, "bgc2": 1}


def test_get_representatives():
    clusters = {
        "bgc1": [["domA", "domB"], ["domC"]],
        "bgc2": [["domA",], ["domC",]],
    }
    graph = nx.Graph()
    graph.add_edges_from([("bgc1", "bgc2")])
    reps = get_representatives(clusters, graph)
    assert len(reps) == 1
    assert "bgc1" in reps


def test_filter_similar_clusters(tmp_path):
    in_file_path = tmp_path / "in_clusters.txt"
    out_file_path = tmp_path / "out_clusters.txt"
    counts_file_path = tmp_path / "counts.txt"
    representatives_file_path = tmp_path / "representatives.txt"
    edges_file_path = tmp_path / "edges.txt"
    sim_cutoff = 0.5
    cores = 1
    verbose = False

    with open(in_file_path, "w") as f:
        f.write("bgc1,domA,domB;domC\n>bgc2\ndomA,domB\n")

    filter_similar_clusters(
        in_file_path,
        out_file_path,
        counts_file_path,
        representatives_file_path,
        edges_file_path,
        sim_cutoff,
        cores,
        verbose,
    )

    with open(out_file_path, "r") as f:
        lines = f.readlines()
        assert len(lines) == 1

    with open(counts_file_path, "r") as f:
        lines = f.readlines()
        assert len(lines) > 0

    with open(representatives_file_path, "r") as f:
        lines = f.readlines()
        assert len(lines) > 0

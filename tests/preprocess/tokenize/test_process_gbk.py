from pathlib import Path

from clusterclue.preprocess.tokenize.process_gbk import convert_gbk2fasta


def test_convert_gbk2fasta(tmp_path):
    # Test the conversion of a GenBank file to a FASTA file
    gbk_path = Path(__file__).parent / "test_data/input/BGC0002659.gbk"
    status = convert_gbk2fasta(
        gbk_file_path=gbk_path,
        out_folder=tmp_path,
        include_contig_edge_clusters=False,
        exclude_name=[],
        verbose=False,
    )
    assert status == "converted"
    expected_fasta_path = (
        Path(__file__).parent / "test_data/expected_output/BGC0002659.fasta"
    )
    out_fasta_path = tmp_path / expected_fasta_path.name
    assert out_fasta_path.read_text() == expected_fasta_path.read_text()

    # Test for already existing FASTA file
    gbk_path = Path(__file__).parent / "test_data/genbank/BGC0002659.gbk"
    out_dir_path = Path(__file__).parent / "test_data/expected_output"
    status = convert_gbk2fasta(
        gbk_file_path=gbk_path,
        out_folder=out_dir_path,
        include_contig_edge_clusters=False,
        exclude_name=[],
        verbose=False,
    )
    assert status == "existed"

    # Test the exclusion of files with certain words in the file name
    gbk_path = Path(__file__).parent / "test_data/input/BGC0002659.gbk"
    status = convert_gbk2fasta(
        gbk_file_path=gbk_path,
        out_folder=tmp_path,
        include_contig_edge_clusters=False,
        exclude_name=["BGC"],
        verbose=False,
    )
    assert status == "excluded"

    # Test the exclusion of contig edge regions
    gbk_path = Path(__file__).parent / "test_data/input/contig_edge_region.gbk"
    status = convert_gbk2fasta(
        gbk_file_path=gbk_path,
        out_folder=tmp_path,
        include_contig_edge_clusters=False,
        exclude_name=[],
        verbose=False,
    )
    assert status == "filtered"

    # Test the inclusion of contig edge region
    status = convert_gbk2fasta(
        gbk_file_path=gbk_path,
        out_folder=tmp_path,
        include_contig_edge_clusters=True,
        exclude_name=[],
        verbose=False,
    )
    assert status == "converted"

    # TODO: Test for parsing errors in the GenBank file.

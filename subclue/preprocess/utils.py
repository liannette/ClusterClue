

def read_txt(in_file_path: str) -> List[str]:
    """Reads a text file into a list of strings, stripping whitespace.

    Args:
        in_file (str): Path to the input file.

    Returns:
        list of str: A list of lines from the file, with leading and trailing whitespace removed.
    """
    with open(in_file_path, "r") as f:
        return [line.strip() for line in f]

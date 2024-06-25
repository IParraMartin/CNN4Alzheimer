import os
import argparse


try:
    import pandas as pd
except:
    print('ImportError: install pandas. Try <pip install pandas> or <pip3 install pandas>')


def get_file_labels(root_dir: str, label: int) -> list:

    """
    Produces tuples of files and labels.

    Parameters: 
        - root directory to all classes in subfolders
        - label (to be assigned in get_items)

    Returns: List of tuples (file, label)
    """

    subfiles = os.listdir(root_dir)
    if not subfiles:
        print('No files found.')
        return []
    return [(file, label) for file in subfiles if file.lower().endswith(('.jpg', '.png', '.jpeg'))]


def get_items(files):

    """
    Produces a flat list of files and labels using get_file_labels function in a loop

    Parameters:
        - files

    Returns: flat list
    """

    temp = []
    for i, file in enumerate(files):
        kind = get_file_labels(file, i)
        temp.append(kind)
    flat_list = [item for sublist in temp for item in sublist]
    return flat_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a .csv file for the __getitem__ in the Dataset class.')
    parser.add_argument('--root_dir', nargs='+', required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--col_names', nargs='+', required=False, default=['filename', 'label'])
    args = parser.parse_args()

    files = args.root_dir

    # Ensure output directory exists
    assert os.path.isdir(args.out_dir), 'Insert a valid directory.'

    # Generate the flat list of file-label pairs
    flat = get_items(files)
    
    df = pd.DataFrame(flat, columns=args.col_names)
    out_file = os.path.join(args.out_dir, 'output.csv')
    df.to_csv(out_file, index=False)

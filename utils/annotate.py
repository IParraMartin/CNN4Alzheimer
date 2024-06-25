import os
try:
    import pandas as pd
except Exception as e:
    print(f'{e}. Install pandas. Try <pip install pandas> or <pip3 install pandas>.')
import argparse


def get_file_labels(root_dir: str, label: int) -> list:

    """
    Produces tuples of files and labels.

    Parameters: 
        - root directory to all classes in subfolders
        - label (to be assigned to each file)

    Returns: List of tuples (file_name, label)
    """

    subfiles = os.listdir(root_dir)

    if not subfiles:
        print(f'No files found in {root_dir}.')
        return []
    
    return [(file, label) for file in subfiles if file.lower().endswith(('.jpg', '.png', '.jpeg'))]


def get_items(root_dir):

    """
    Produces a flat list of files and labels using get_file_labels function in a loop

    Parameters:
        - root_dir: Root directory containing subdirectories

    Returns: flat list
    """

    temp = []
    subdirectories = [os.path.join(root_dir, sub_dir) for sub_dir in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, sub_dir))]
    
    for i, sub_dir in enumerate(subdirectories):
        kind = get_file_labels(sub_dir, i)
        temp.append(kind)
    
    flat_list = [item for sublist in temp for item in sublist]

    return flat_list


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Create a .csv file for the __getitem__ in the Dataset class.')
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--col_names', nargs='+', required=False, default=['filename', 'label'])
    args = parser.parse_args()

    root_dir = args.root_dir
    out_dir = args.out_dir

    assert os.path.isdir(out_dir), 'Insert a valid output directory.'

    flat = get_items(root_dir)

    df = pd.DataFrame(flat, columns=args.col_names)
    out_file = os.path.join(out_dir, 'output.csv')
    df.to_csv(out_file, index=False)
    print(f'CSV file created at {out_file}')

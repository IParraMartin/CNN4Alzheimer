import os
try:
    import pandas as pd
except Exception as e:
    print(f'An error ocurred: {e}.\nTry <pip install pandas> or <pip3 install pandas>')
import argparse


def get_file_labels(path, label):
    filenames = os.listdir(path)
    if not filenames:
        print('No file path found.')
        return []
    return [(file, label) for file in filenames if file.lower().endswith(('.jpg', '.png', '.jpeg'))]

def get_items(files):
    temp = []
    for i, file in enumerate(files):
        kind = get_file_labels(file, i)
        temp.append(kind)
    flat_list = [item for sublist in temp for item in sublist]
    return flat_list


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Create a .csv file for the __getitem__ in the Dataset class.')
    parser.add_argument('--files_dirs', nargs='+', required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--col_names', nargs='+', required=False)
    args = parser.parse_args()

    files = args.files_dirs

    # FILES = [
    #     'Dataset/Non_Demented', 
    #     'Dataset/Very_Mild_Demented', 
    #     'Dataset/Mild_Demented', 
    #     'Dataset/Moderate_Demented'
    # ]

    if args.out_path:
        assert os.path.isdir(args.out_path), 'Insert a valid directory.'

        flat = get_items(files)
        df = pd.DataFrame(flat, columns=args.col_names)
        df.to_csv(args.out_dir)
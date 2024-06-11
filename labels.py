import os
import pandas as pd

FILES = [
    'Dataset/Non_Demented', 
    'Dataset/Very_Mild_Demented', 
    'Dataset/Mild_Demented', 
    'Dataset/Moderate_Demented'
]

SAVE_PATH = '/Users/inigoparra/Desktop/Run2Recommend/labels.csv'

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
    
    flat = get_items(FILES)
    df = pd.DataFrame(flat, columns=['name', 'label'])
    df.to_csv(SAVE_PATH)
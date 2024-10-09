import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import re
from torch.utils.data import DataLoader
import torch


class AGUDataset(Dataset):
    def __init__(self, txt_dir, label_name, label_names={'etot': 'Etot (eV/atom)', 'emix': 'Emix (eV/atom)', 'eform': 'Eform (eV/atom)', 'ms': 'Ms (mub/atom)', 'mb': 'mb (mub/cell)', 'rmsd': 'rmsd (\\AA)'}):
        """
        Initializes the dataset.

        :param txt_dir: Directory where .txt files are stored.
        :param label_name: Path to the Excel file containing labels.
        :param label_names: Dictionary of label names to fetch specific labels.
        """
        self.txt_dir = txt_dir
        self.label_name = label_name
        self.label_names = label_names
        self.labels = self.read_excel()


    def read_excel(self):
        """
        Reads the Excel file containing labels.

        :return: A dictionary mapping file names to their properties.
        """
        wb = pd.read_excel(self.label_name, engine='openpyxl')
        labels = {}
        for index, row in wb.iterrows():
            file_name = row['structures']  # Adjust column name as needed
            labels[file_name] = {k: row[k] for k in row.index if k in self.label_names.values()}
        return labels

    def __getitem__(self, index):
        """
        Retrieves an item by its index.

        :param index: Index of the item.
        :return: A tuple of the one-dimensional data and its specific labels.
        """
        file_name = list(self.labels.keys())[index]
        file_path = os.path.join(self.txt_dir, file_name + '.txt')

        with open(file_path, 'r') as file:
            content = file.read()
            numbers = re.findall(r'[-+]?\d*\.\d+|\d+', content)
            data = np.array([float(num) for num in numbers], dtype=np.float32)


        labels = self.labels[file_name]
        # Fetch only the labels specified by the label names
        selected_labels = {key: labels[value] for key, value in self.label_names.items() if value in labels}

        return data, selected_labels

    def __len__(self):
        return len(self.labels)

if __name__ == '__main__':
    dataset = AGUDataset(txt_dir='../HEA_Data/change_fcc',
                         label_name='../HEA_Data/Out_labels/Database_fcc.xlsx')

    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    for data, labels in data_loader:
        task='etot'
        data=data.to('cuda')
        labels_on_gpu = {key: torch.tensor(value, dtype=torch.float32).to('cuda') for key, value in labels.items()}
        print(labels_on_gpu)
        print(data)
        print(data.device, data.dtype, labels_on_gpu['etot'].dtype, data.shape, labels_on_gpu['etot'].shape)
    # df = pd.read_excel('../HEA_Data/Out_labels/data.xlsx', engine='openpyxl')
    # df_fcc = df[df['structures'].str.contains('fcc', case=False)]
    # df_bcc = df[df['structures'].str.contains('bcc', case=False)]
    #
    # df_fcc.reset_index(drop=True, inplace=True)
    # df_bcc.reset_index(drop=True, inplace=True)
    # df_fcc.index += 0
    # df_bcc.index += 0
    #
    # df_fcc.to_excel('../HEA_Data/Out_labels/Database_fcc.xlsx', index=True)
    # df_bcc.to_excel('../HEA_Data/Out_labels/Database_bcc.xlsx', index=True)
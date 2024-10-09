import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import re
from torch.utils.data import DataLoader
import torch


class GANDataset(Dataset):
    def __init__(self, txt_dir, label_name, label_names={'Cr': 'Cr', 'Mn': 'Mn', 'Fe': 'Fe', 'Co': 'Co', 'Ni': 'Ni', 'Pd': 'Pd'}):
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
        selected_labels_values = list(selected_labels.values())
        selected_labels_np = np.array(selected_labels_values, dtype=np.float32)

        return data, selected_labels_np

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    dataset = GANDataset(txt_dir='../change_fcc',
                         label_name='../change_fcc/fcc.xlsx')

    data_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=False)

    for data, labels in data_loader:
        data=data.to('cuda')
        labels= labels.to('cuda')
        print(labels)
        print(data)
        print(data.device, data.dtype, labels.dtype, data.shape, labels.shape)

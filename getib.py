import copy
from typing import Tuple

import numpy as np
from torch_geometric.nn.models import SchNet, DimeNet
from torch_geometric.loader import DataLoader
from datasets.HEA_dataset import HEADataset
import torch
from utils.meter import mae
import torch.nn as nn
from torch import Tensor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from models.heanet import HeaNet
import argparse
from utils.registry import registry, setup_imports
import time
from datasets.Mp_dataset import MpDataset, load_dataset, MpGeometricDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import SequentialSampler
from utils.utility import DataTransformer
import os
from trainer_heanet_mtl import mtl_criterion, evaluate
import pandas as pd

data_transform = DataTransformer()
device = torch.device(device='cuda' if torch.cuda.is_available() else 'cpu')

current_file_dir = os.path.dirname(__file__)
data_dir = os.path.join(current_file_dir, 'HEA_Data/POSCARS')


def load_hea_original_data():
    """
    Load the HEA dataset from the original data source.
    :return:
    """
    data_file = ''
    data_file = os.path.normpath(os.path.join(current_file_dir, 'HEA_Data/Out_labels/Database.xlsx'))
    print('loading (4+5) component HEA data with randomly splitting all the data into training/testing set\t'
          'from {}.'.format(data_file))
    total_dataset = HEADataset(poscar_dir=data_dir, label_name=data_file)
    print(total_dataset)
    total_loder = DataLoader(dataset=total_dataset, batch_size=args.batch_size, shuffle=False)
    return total_loder


def getinteractions(model, loader, tasks):
    out_types = []
    out_z = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch.to(device)
            z, types = model(batch.atomic_numbers.long(), batch.pos, batch=batch, get_ib=True)
            types = types.squeeze()
            assert isinstance(tasks, list), ' task parameter must be a list in the multi-task learning cases.'
            out_z.append(z.cpu().numpy())
            out_types.append(types.cpu().numpy())
    return out_z, out_types


def printinteractions():
    model_name = './saved_models_mtl_HEA/mtl_3_etot_emix_eform_HEA_500_b0_best.pt'
    model_state = torch.load(model_name)
    model.load_state_dict(model_state)
    out_z, out_types = getinteractions(model, total_loder, tasks=args.task)
    data_file = os.path.normpath(os.path.join(current_file_dir, 'HEA_Data/Out_labels/Database.xlsx'))
    df = pd.read_excel(data_file)
    hea_name=df.iloc[:, 1]
    if not os.path.exists('ori_ib'):
        os.makedirs('ori_ib')
    if not os.path.exists('change_fcc'):
        os.makedirs('change_fcc')
    if not os.path.exists('change_bcc'):
        os.makedirs('change_bcc')
    fcc_label = []
    bcc_label = []
    for i in range(len(out_types)):
        # Create a valid filename by replacing spaces with underscores
        change_types=[]
        label_types = []

        zero_array = np.zeros(128)
        filename = hea_name[i].replace(" ", "_") + ".txt"
        # Define the path to the file
        filepath = os.path.join('ori_ib', filename)
        np.savetxt(filepath, out_types[i], fmt='%f')
        unique_elements, counts = np.unique(out_z[i], return_counts=True)
        percentages = counts / out_z[i].size
        # print(filename)
        # for element, percentage in zip(unique_elements, percentages):
        #     print(f"Element: {element}, Percentage: {percentage}")
        j=0
        label_types.append(hea_name[i])
        if 24 in unique_elements:
            change_types.append(out_types[i][j])
            index = np.where(unique_elements == 24)[0][0]
            percentage = percentages[index]
            label_types.append(percentage)
            j=j+1
        else:
            change_types.append(zero_array)
            label_types.append(0.)
        if 25 in unique_elements:
            change_types.append(out_types[i][j])
            index = np.where(unique_elements == 25)[0][0]
            percentage = percentages[index]
            label_types.append(percentage)
            j=j+1
        else:
            change_types.append(zero_array)
            label_types.append(0.)
        if 26 in unique_elements:
            change_types.append(out_types[i][j])
            index = np.where(unique_elements == 26)[0][0]
            percentage = percentages[index]
            label_types.append(percentage)
            j=j+1
        else:
            change_types.append(zero_array)
            label_types.append(0.)
        if 27 in unique_elements:
            change_types.append(out_types[i][j])
            index = np.where(unique_elements == 27)[0][0]
            percentage = percentages[index]
            label_types.append(percentage)
            j=j+1
        else:
            change_types.append(zero_array)
            label_types.append(0.)
        if 28 in unique_elements:
            change_types.append(out_types[i][j])
            index = np.where(unique_elements == 28)[0][0]
            percentage = percentages[index]
            label_types.append(percentage)
            j=j+1
        else:
            change_types.append(zero_array)
            label_types.append(0.)
        if 46 in unique_elements:
            change_types.append(out_types[i][j])
            index = np.where(unique_elements == 46)[0][0]
            percentage = percentages[index]
            label_types.append(percentage)
            j=j+1
        else:
            change_types.append(zero_array)
            label_types.append(0.)
        if "fcc" in filename:
            filepath = os.path.join('change_fcc', filename)
            fcc_label.append(label_types)
        elif "bcc" in filename:
            filepath = os.path.join('change_bcc', filename)
            bcc_label.append(label_types)
        else:
            continue
        np.savetxt(filepath, change_types, fmt='%f')
        print(filename)
        print(label_types)
    df = pd.DataFrame(fcc_label, columns=['structures', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Pd'])
    with open('change_fcc/fcc.xlsx', 'w') as f:
        pass
    df.to_excel('change_fcc/fcc.xlsx', index=True)
    df = pd.DataFrame(bcc_label, columns=['structures', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Pd'])
    with open('change_bcc/bcc.xlsx', 'w') as f:
        pass
    df.to_excel('change_bcc/bcc.xlsx', index=True)

    # print(len(out_z))
    # print(len(out_types))
    # print(len(out_types[0]))
    # data = np.loadtxt('ori_ib/FeNiCoCr_sqsfcc.txt')
    # print(data)
    return out_types


if __name__ == '__main__':
    # arguments for settings.
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_channels', type=int, default=128, help='number of hidden channels for the embeddings')
    parser.add_argument('--n_filters', type=int, default=64, help='number of filters')
    parser.add_argument('--n_interactions', type=int, default=3, help='number of interaction blocks')
    parser.add_argument('--n_gaussian', type=int, default=50, help='number of gaussian bases to expand the distances')
    parser.add_argument('--cutoff', type=float, default=10, help='the cutoff radius to consider passing messages')
    parser.add_argument('--aggregation', type=str, default='add', help='the aggregation scheme ("add", "mean", or "max") \
                                                                       to use in the messages')
    parser.add_argument('--seed', type=int, default=1454880, help='random seed')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs to train the model')
    parser.add_argument('--lr', type=float, default=1e-3, help='the learning rate of the algorithm')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--task', type=str, default=[], nargs='+', help='the target to model in the dataset')
    parser.add_argument('--model', type=str, default='heanet', help='the name of the ML model')
    parser.add_argument('--scaling', type=float, default=1000, help='the scaling factor of the target.')
    parser.add_argument('--batch_size', type=int, default=16, help='the batch size of the data loader')
    parser.add_argument('--save_model', type=bool, default=True, help='save the trained model')
    parser.add_argument('--transform', type=str, default=[], nargs='+',
                        help='transform the target proerty, only support log and scaling')
    parser.add_argument('--tower_h1', type=int, default=128,
                        help='validate the model during training the ef and eg tasks')
    parser.add_argument('--tower_h2', type=int, default=64,
                        help='validate the model during training the ef and eg tasks')
    parser.add_argument('--use_pbc', action='store_true', help='Whether to use the periodic boundary condition.')
    parser.add_argument('--train', '-t', action='store_true', help='Training the ECMTL model.')
    parser.add_argument('--predict', '-p', action='store_true', help='Applying the ECMTL to predict something.')
    parser.add_argument('--fine_tune', '-f', action='store_true', help='fine tune the ECMTL model to predict HEAs.')
    parser.add_argument('--is_validate', action='store_true',
                        help='validate the model during training the ef and eg tasks')
    parser.add_argument('--split_type', type=int, default=0, help='the splitting type of the dataset')
    parser.add_argument('--processed_data', action='store_true',
                        help='whether to load the preprocessed data according to number of components.')
    # parser.add_argument('--saved_model', type='str', default='etot_type0_200.pt', help='the trained model')

    args = parser.parse_args()

    setup_imports()
    RANDOM_SEED = args.seed  # 1454880
    is_validate = args.is_validate
    is_processed = args.processed_data
    num_epochs = args.epochs
    scaling = args.scaling
    model = registry.get_model(args.model
                               )(hidden_channels=args.hidden_channels,
                                 num_filters=args.n_filters,
                                 num_interactions=args.n_interactions,
                                 num_gaussians=args.n_gaussian,
                                 cutoff=args.cutoff,
                                 readout=args.aggregation,
                                 dipole=False, mean=None, std=None,
                                 atomref=None, num_tasks=len(args.task),
                                 tower_h1=args.tower_h1,
                                 tower_h2=args.tower_h2,
                                 use_pbc=args.use_pbc,
                                 )
    model.to(device)

    if args.predict:
        # python trainer_heanet_mtl_HEA.py --task etot emix eform  --batch_size 128  --is_validate  --split_type 2 -p
        total_loder = load_hea_original_data()
        out_types = printinteractions()



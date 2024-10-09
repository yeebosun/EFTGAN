import os
import warnings
import os.path as osp
from math import pi as PI

import ase
import torch
import torch.nn.functional as F
from torch.nn import Embedding, Sequential, Linear, ModuleList, BatchNorm1d, ELU, ReLU
import numpy as np

from torch_scatter import scatter
from torch_geometric.data.makedirs import makedirs
from torch_geometric.data import download_url, extract_zip
from torch_geometric.nn import radius_graph, MessagePassing
from utils.registry import registry
from utils.utility import get_pbc_distances, radius_graph_pbc

try:
    import schnetpack as spk
except ImportError:
    spk = None

qm9_target_dict = {
    0: 'dipole_moment',
    1: 'isotropic_polarizability',
    2: 'homo',
    3: 'lumo',
    4: 'gap',
    5: 'electronic_spatial_extent',
    6: 'zpve',
    7: 'energy_U0',
    8: 'energy_U',
    9: 'enthalpy_H',
    10: 'free_energy',
    11: 'heat_capacity',
}


@registry.register_models('agunet')
class AguNet(torch.nn.Module):

    def __init__(self, hidden_channels=128, num_filters=128,
                 num_interactions=6, num_gaussians=50, cutoff=10.0,
                 readout='add', dipole=False, mean=None, std=None,
                 atomref=None, num_tasks=1, tower_h1=128, tower_h2=64,
                 use_pbc=False, n_seq=None):
        super(AguNet, self).__init__()

        assert readout in ['add', 'sum', 'mean']

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.readout = readout
        self.dipole = dipole
        self.readout = 'add' if self.dipole else self.readout
        self.mean = mean
        self.std = std
        self.scale = None
        self.num_tasks = num_tasks
        self.tower_h1 = tower_h1
        self.tower_h2 = tower_h2
        self.use_pbc = use_pbc
        self.n_seq = n_seq
        self.tower_layers = ModuleList()
        self.tower_heads = ModuleList()
        self.task_heads = ModuleList()
        self.tower_lin1 = Linear(30, self.tower_h1)
        self.tower_lin2 = Linear(self.tower_h1, self.tower_h2)
        self.relu = torch.nn.ReLU()

        for _ in range(self.num_tasks):
            tower = Sequential(
                self.tower_lin1,
                ShiftedSoftplus(),
                # BatchNorm1d(self.tower_h1,affine=False),
                self.tower_lin2,
                ShiftedSoftplus(),
                # BatchNorm1d(self.tower_h2, affine=False),
                Linear(self.tower_h2, 1)
            )
            self.tower_layers.append(tower)

        self.lin1 = Linear(5, self.tower_h1)
        # self.act1 = self.relu()
        self.lin2 = Linear(self.tower_h1, self.tower_h2)
        # self.act2 = self.relu()
        self.lin3 = Linear(self.tower_h2, 1)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin3.weight)
        self.lin3.bias.data.fill_(0)

    def forward(self, x):

        if self.num_tasks > 1:
            outs = []
            print(x.shape)
            for i in range(self.num_tasks):
                out_i = self.tower_layers[i](x)
                outs.append(out_i)
            out = outs
            print(out)
        else:
            out = []
            outs = self.lin1(x)
            outs = self.relu(outs)
            outs = self.lin2(outs)
            outs = self.relu(outs)
            outs = self.lin3(outs)
            out.append(outs)

        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'hidden_channels={self.hidden_channels}, '
                f'num_filters={self.num_filters}, '
                f'num_interactions={self.num_interactions}, '
                f'num_gaussians={self.num_gaussians}, '
                f'cutoff={self.cutoff})')





class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


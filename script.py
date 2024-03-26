"""
data preparation for example data qm9
""" 

try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
import copy
import utils
import argparse
import wandb
from configs.datasets_config import get_dataset_info
from os.path import join
from qm9 import dataset
# from qm9.models import get_optim, get_model
# from equivariant_diffusion import en_diffusion
# from equivariant_diffusion.utils import assert_correctly_masked
# from equivariant_diffusion import utils as flow_utils
import torch
import torch.nn as nn
import time
import pickle
from qm9.utils import prepare_context, compute_mean_mad
import sys
from tqdm import tqdm
# from train_test import train_epoch, test, analyze_and_save
print("all imports done")

# def parse_arguments(argument_string):
#     parser = argparse.ArgumentParser(description='Process arguments for data loading.')

#     # Define the expected arguments
#     parser.add_argument('--n_epochs', type=int)
#     parser.add_argument('--exp_name', type=str)
#     parser.add_argument('--n_stability_samples', type=int)
#     parser.add_argument('--diffusion_noise_schedule', type=str)
#     parser.add_argument('--diffusion_noise_precision', type=float)
#     parser.add_argument('--diffusion_steps', type=int)
#     parser.add_argument('--diffusion_loss_type', type=str)
#     parser.add_argument('--batch_size', type=int)
#     parser.add_argument('--nf', type=int)
#     parser.add_argument('--n_layers', type=int)
#     parser.add_argument('--lr', type=float)
#     parser.add_argument('--normalize_factors', type=eval)  # using eval can be risky; ensure the input is safe
#     parser.add_argument('--test_epochs', type=int)
#     parser.add_argument('--ema_decay', type=float)
    
#     # Simulate command-line argument passing by splitting the string and then parsing
#     # args = parser.parse_args(argument_string.split())
#     args = parser.parse_args()
    
#     return args

parser = argparse.ArgumentParser(description='E3Diffusion')
parser.add_argument('--exp_name', type=str, default='debug_10')
parser.add_argument('--model', type=str, default='egnn_dynamics',
                    help='our_dynamics | schnet | simple_dynamics | '
                         'kernel_dynamics | egnn_dynamics |gnn_dynamics')
parser.add_argument('--probabilistic_model', type=str, default='diffusion',
                    help='diffusion')

# Training complexity is O(1) (unaffected), but sampling complexity is O(steps).
parser.add_argument('--diffusion_steps', type=int, default=500)
parser.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2',
                    help='learned, cosine')
parser.add_argument('--diffusion_noise_precision', type=float, default=1e-5,
                    )
parser.add_argument('--diffusion_loss_type', type=str, default='l2',
                    help='vlb, l2')

parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--brute_force', type=eval, default=False,
                    help='True | False')
parser.add_argument('--actnorm', type=eval, default=True,
                    help='True | False')
parser.add_argument('--break_train_epoch', type=eval, default=False,
                    help='True | False')
parser.add_argument('--dp', type=eval, default=True,
                    help='True | False')
parser.add_argument('--condition_time', type=eval, default=True,
                    help='True | False')
parser.add_argument('--clip_grad', type=eval, default=True,
                    help='True | False')
parser.add_argument('--trace', type=str, default='hutch',
                    help='hutch | exact')
# EGNN args -->
parser.add_argument('--n_layers', type=int, default=6,
                    help='number of layers')
parser.add_argument('--inv_sublayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--nf', type=int, default=128,
                    help='number of layers')
parser.add_argument('--tanh', type=eval, default=True,
                    help='use tanh in the coord_mlp')
parser.add_argument('--attention', type=eval, default=True,
                    help='use attention in the EGNN')
parser.add_argument('--norm_constant', type=float, default=1,
                    help='diff/(|diff| + norm_constant)')
parser.add_argument('--sin_embedding', type=eval, default=False,
                    help='whether using or not the sin embedding')
# <-- EGNN args
parser.add_argument('--ode_regularization', type=float, default=1e-3)
parser.add_argument('--dataset', type=str, default='qm9',
                    help='qm9 | qm9_second_half (train only on the last 50K samples of the training dataset)')
parser.add_argument('--datadir', type=str, default='qm9/temp',
                    help='qm9 directory')
parser.add_argument('--filter_n_atoms', type=int, default=None,
                    help='When set to an integer value, QM9 will only contain molecules of that amount of atoms')
parser.add_argument('--dequantization', type=str, default='argmax_variational',
                    help='uniform | variational | argmax_variational | deterministic')
parser.add_argument('--n_report_steps', type=int, default=1)
parser.add_argument('--wandb_usr', type=str)
parser.add_argument('--no_wandb', action='store_true', help='Disable wandb')
parser.add_argument('--online', type=bool, default=True, help='True = wandb online -- False = wandb offline')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--save_model', type=eval, default=True,
                    help='save model')
parser.add_argument('--generate_epochs', type=int, default=1,
                    help='save model')
parser.add_argument('--num_workers', type=int, default=0, help='Number of worker for the dataloader')
parser.add_argument('--test_epochs', type=int, default=10)
parser.add_argument('--data_augmentation', type=eval, default=False, help='use attention in the EGNN')
parser.add_argument("--conditioning", nargs='+', default=[],
                    help='arguments : homo | lumo | alpha | gap | mu | Cv' )
parser.add_argument('--resume', type=str, default=None,
                    help='')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='')
parser.add_argument('--ema_decay', type=float, default=0.999,
                    help='Amount of EMA decay, 0 means off. A reasonable value'
                         ' is 0.999.')
parser.add_argument('--augment_noise', type=float, default=0)
parser.add_argument('--n_stability_samples', type=int, default=500,
                    help='Number of samples to compute the stability')
parser.add_argument('--normalize_factors', type=eval, default=[1, 4, 1],
                    help='normalize factors for [x, categorical, integer]')
parser.add_argument('--remove_h', action='store_true')
parser.add_argument('--include_charges', type=eval, default=True,
                    help='include atom charge or not')
parser.add_argument('--visualize_every_batch', type=int, default=1e8,
                    help="Can be used to visualize multiple times per epoch")
parser.add_argument('--normalization_factor', type=float, default=1,
                    help="Normalize the sum aggregation of EGNN")
parser.add_argument('--aggregation_method', type=str, default='sum',
                    help='"sum" or "mean"')
args = parser.parse_args()


# Your arguments string
"""
--n_epochs 3000 --exp_name edm_qm9 --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 20 --ema_decay 0.9999
"""

###### Utilities 
def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)
def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask)).abs().max().item() < 1e-4, \
        'Variables not masked properly.'


###### losses
def compute_loss_and_nll(args, generative_model, nodes_dist, x, h, node_mask, edge_mask, context):
    bs, n_nodes, n_dims = x.size()


    if args.probabilistic_model == 'diffusion':
        edge_mask = edge_mask.view(bs, n_nodes * n_nodes)

        assert_correctly_masked(x, node_mask)

        # Here x is a position tensor, and h is a dictionary with keys
        # 'categorical' and 'integer'.
        nll = generative_model(x, h, node_mask, edge_mask, context)

        N = node_mask.squeeze(2).sum(1).long()

        log_pN = nodes_dist.log_prob(N)

        assert nll.size() == log_pN.size()
        nll = nll - log_pN

        # Average over batch.
        nll = nll.mean(0)

        reg_term = torch.tensor([0.]).to(nll.device)
        mean_abs_z = 0.
    else:
        raise ValueError(args.probabilistic_model)

    return nll, reg_term, mean_abs_z

### model 

# graph convolution layer
class GCL(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=0, nodes_att_dim=0, act_fn=nn.SiLU(), attention=False):
        """GRAPH CONVOLUTION LAYERS 
        Basic gnn convoltion (nodes, edges) -> returning h (graph features) and m_ij (edge features)

        Args:
            input_nf (_type_): dimensionality of the input node features
            output_nf (_type_): dimensionality of the output node features
            hidden_nf (_type_): dimensionality of the hidden layer
            normalization_factor (_type_): to normalize the aggregated node features
            aggregation_method (_type_): _description_
            edges_in_d (int, optional): _description_. Defaults to 0.
            nodes_att_dim (int, optional): _description_. Defaults to 0.
            act_fn (_type_, optional): _description_. Defaults to nn.SiLU().
            attention (bool, optional): _description_. Defaults to False.
        """
        super(GCL, self).__init__()
        input_edge = input_nf * 2 # edges = nodes in pairs 
        self.normalization_factor = normalization_factor # nomalization factor for aggregated features
        self.aggregation_method = aggregation_method # aggregation method to produce graph features (graph representation)
        self.attention = attention

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    # Message passing for GCN
    """
    In GCN, features of the source (neighbors) and target (node) are concatenated and passed through a multi-layer perceptron (MLP) 
    to produce the edge features.This is a common operation in graph neural networks where information from neighboring nodes (source nodes) 
    is aggregated to update the features of the current node (target node).
    
    The reason for concatenating these two is to combine the 
    information from both the source and target nodes into a single tensor, which can then be processed further.This is part of the 
    message passing framework in GCNs where each node sends a message (its features) to its neighbors, and these messages are aggregated
    and used to update the node's own features.

    """
    def edge_model(self, source, target, edge_attr, edge_mask): 
        """_summary_

        Args:
            source (_type_): neighbors' edge features (the degree??)
            target (_type_): node's edge features   
            edge_attr (_type_): edge features
            edge_mask (_type_): optional edge mask, used to mask out edges (set some edges to zero, e.g. to ignore them during training). Defaults to None.

        Returns:
            _type_: _description_
        """
        # Concatenating source and target edge features (message passing in GCN)
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target], dim=1) 
        else:
            out = torch.cat([source, target, edge_attr], dim=1)
        mij = self.edge_mlp(out)

        if self.attention:
            att_val = self.att_mlp(mij)
            out = mij * att_val
        else:
            out = mij

        if edge_mask is not None:
            out = out * edge_mask
        return out, mij

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = x + self.node_mlp(agg) # aggregated information from nodes, edges, and node attributes
        return out, agg

    def forward(self, h, edge_index, edge_attr=None, node_attr=None, node_mask=None, edge_mask=None): # h0 is the atoms types and charges
        row, col = edge_index
        edge_feat, mij = self.edge_model(h[row], h[col], edge_attr, edge_mask)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        if node_mask is not None:
            h = h * node_mask
        return h, mij # h is the node features, mij is the edge features


# GNN model 

class GNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, aggregation_method='sum', device='cpu',
                 act_fn=nn.SiLU(), n_layers=4, attention=False,
                 normalization_factor=1, out_node_nf=None):
        super(GNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        ### Encoder
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(
                self.hidden_nf, self.hidden_nf, self.hidden_nf,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
                edges_in_d=in_edge_nf, act_fn=act_fn,
                attention=attention))
        self.to(self.device)

    def forward(self, h, edges, edge_attr=None, node_mask=None, edge_mask=None):
        # Edit Emiel: Remove velocity as input
        h = self.embedding(h) # graph node embedding (initialize the embedding)
        
        # graph convolution layers (number of layers)
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edges, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
        h = self.embedding_out(h)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h






### Load the dataset
dataloaders, charge_scale = dataset.retrieve_dataloaders(args)

print(dataloaders)
print(charge_scale)
data_dummy = next(iter(dataloaders['train']))
# one batch of data
print(data_dummy)
with open('./exp_data/data_dummy.pkl', 'wb') as f:
    pickle.dump(data_dummy, f)
# print(data_dummy.shape)

# Write training loop (just for experimenting)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epoch = 1

dp = True # data parallelism 

# model 
model, nodes_dist, prop_dist = get_model(args, device, dataset_info, dataloaders['train'])

model = model.to(device)


# Initialize dataparallel if enabled and possible.
    if dp and torch.cuda.device_count() > 1:
        print(f'Training using {torch.cuda.device_count()} GPUs')
        model_dp = torch.nn.DataParallel(model.cpu())
        model_dp = model_dp.cuda()
    else:
        model_dp = model



# Training loop 
for epoch in range(num_epoch):
    with tqdm(dataloaders['train'], unit='batch') as tepoch:
        for idx_batch, data in enumerate(dataloaders['train']):
            x = data['positions'].to(device)
            node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
            edge_mask = data['edge_mask'].to(device, dtype)
            one_hot = data['one_hot'].to(device, dtype)
            charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)

            check_mask_correct([x, one_hot, charges], node_mask) # check the mask of the is correctly reflecting the padded nodes
            
            h = {'categorical': one_hot, 'integer': charges}
            
            # nll 
            nll, reg_term, mean_abs_z = compute_loss_and_nll(args, model_dp, nodes_dist,
                                                                x, h, node_mask, edge_mask, context)
            
            




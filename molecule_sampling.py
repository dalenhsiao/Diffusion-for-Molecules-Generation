import torch
import numpy as np
import argparse
import os
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader


@torch.no_grad()
def sampling(model, diffusion, data_shape, edge_index):
    model.eval()
    generated_sample = diffusion.sample(data_shape, edge_index)
    generated_sample = generated_sample.detach().cpu().numpy()
    return generated_sample

def convert_prob_dist_to_atom_class(data: np.array):
    labels = ['H', 'C', 'N', 'O', 'F']
    atoms = np.array([labels[ind] for ind in np.argmax(data, axis=1)])
    return atoms

def save_sample(data: np.array, adj_mat, gt: np.array, root: str, sample_num: int):
    if os.path.exists(root):
        pass
    else:
        # Ensure the directory exists
        os.makedirs(root, exist_ok=True)
    mole_file = os.path.join(root, f"sample{sample_num}.txt")
    adj_file = os.path.join(root, f"adj_matrix_sample{sample_num}.txt")
    atoms = convert_prob_dist_to_atom_class(data)
    gt_labels = convert_prob_dist_to_atom_class(gt)
    output = np.column_stack((data, atoms, gt_labels))
    np.savetxt(mole_file, output, delimiter=' ', fmt='%s', comments='', header='H C N O F Pred GT')
    np.savetxt(adj_file, adj_mat, delimiter=' ', fmt='%s', comments='')

def sample_qm9(num_sample):
    data = QM9(root='./qm9_data', transform=None)
    dataloader = DataLoader(data, batch_size=1, shuffle=True)
    sampled_data = []
    samples_collected = 0

    # Iterate through batches in the dataloader
    for batch in dataloader:
        # Add the batch (or the portion of it needed) to the list of sampled data
        sampled_data.append(batch)
        samples_collected += len(batch.idx)

        # If we've collected enough samples, break out of the loop
        if samples_collected >= num_sample:
            break

    # Depending on your use case, you may want to return a concatenated version of the samples
    # or you might keep the list of batches if further batch processing is required.
    return sampled_data

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=str, default="configs")
    args = parser.parse_args()
    args_dict = vars(args)
    return args_dict


"""
python molecule_sampling.py --configs configs
"""

"""
Generate a set of molecules that takes in the edge index of a given dataset
Using the qm9 dataset (or other dataset) as provided source of edge index
number of nodes (atoms) and edge index are given, then generate molecule samples

"""


if __name__ == "__main__":
    from DiffuseSampler import DiffusionModel
    from base_model import *
    from omegaconf import OmegaConf
    
    
    args_dict = arg_parse()
    config = OmegaConf.load(f'{args_dict["configs"]}.yaml')
    config = config.sampling
    models_path = "diffusion_models"
    pth = os.path.join(models_path, f"{config.load_model_param}.pth")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random_sample = sample_qm9(config.num_sample)
    data_shape = [(sample.x.shape[0], 5) for sample in random_sample] # one single molecule sample
    sample_fp = os.path.join("sampling", f"{config.experiment_run}")
    # import pdb ; pdb.set_trace()
    # GNN model
    net = GNN(
        n_feat_in=5,
        layers=config.layers,
        latent_space_dims=config.latent_space_dims,
        time_emb_dim=4,
        fine_tune=True
        ).to(device)
    net.load_state_dict(
        torch.load(pth)
    )
    # diffusion model
    diffusion = DiffusionModel(
        net,
        timesteps=config.diffuse_timesteps,
    ).to(device)
    # Generating sample
    for idx, sample in enumerate(random_sample):
        gen_sample = sampling(net, diffusion, data_shape[idx], sample.edge_index.to(device))
        # import pdb ; pdb.set_trace()
        gt = np.array(sample.x[:,:5])
        save_sample(gen_sample, sample.edge_index, gt, sample_fp, idx)
        print("sample created")

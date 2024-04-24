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

def save_sample(data, root, sample_num: int):
    if os.path.exists(root):
        pass
    else:
        # Ensure the directory exists
        os.makedirs(root, exist_ok=True)
    mole_file = os.path.join(root, f"sample{sample_num}.txt")
    np.savetxt(mole_file, data, delimiter=' ', fmt='%d', header='H C N O F')

def sample_qm9(num_sample):
    data = QM9(root='./practice_data', transform=None)
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
    parser.add_argument("--experiment", "-exp", type=str, default="gnn")
    parser.add_argument("--experiment_run", "-r", type=str, default="experiement_run")
    parser.add_argument("--load_model_param", type=str, default="models.pth")
    
    parser.add_argument("--num_sample", type=int, default=10)
    parser.add_argument("--diffuse_timesteps", "-tsteps", type=int, default=100)
    parser.add_argument("--batch_size", "-bs", type=int, default=64)
    parser.add_argument("--layers", nargs='+', type=int, default=[32, 64, 128])
    args = parser.parse_args()
    args_dict = vars(args)
    return args_dict


"""
python molecule_sampling.py --experiment molecule_generation --experiment_run diffusion_model_fine_tuned --num_sample 10 --load_model_param diffusion_model_fine_tuned --diffuse_timesteps 100 --layers 32 64 128
"""

"""
Generate a set of molecules that takes in the edge index of a given dataset
Using the qm9 dataset (or other dataset) as provided source of edge index
number of nodes (atoms) and edge index are given, then generate molecule samples

"""


if __name__ == "__main__":
    from DiffuseSampler import DiffusionModel
    from based_model_4_4 import Net
    
    
    args_dict = arg_parse()
    models_path = "diffusion_models"
    pth = os.path.join(models_path, f"{args_dict["load_model_param"]}.pth")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random_sample = sample_qm9(args_dict["num_sample"])
    data_shape = [(sample.x.shape[0], 5) for sample in random_sample] # one single molecule sample
    sample_fp = os.path.join("sampling", f"{args_dict["experiment_run"]}")
    # import pdb ; pdb.set_trace()
    # GNN model
    net = Net(
        n_feat_in=5,
        layers=args_dict["layers"],
        time_emb_dim=4,
        fine_tune=True
        ).to(device)
    net.load_state_dict(
        torch.load(pth)
    )
    # diffusion model
    diffusion = DiffusionModel(
        net,
        timesteps=args_dict["diffuse_timesteps"],
    ).to(device)
    # Generating sample
    for idx, sample in enumerate(random_sample):
        gen_sample = sampling(net, diffusion, data_shape[idx], sample.edge_index.to(device))
        save_sample(gen_sample, sample_fp, idx)
        print("sample created")

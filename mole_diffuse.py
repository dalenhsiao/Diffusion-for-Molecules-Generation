# loading the pretrained model into diffusion
import argparse
def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=str, default="configs")
    args = parser.parse_args()
    args_dict = vars(args)
    return args_dict


if __name__ == "__main__":
    # Test encoder
    from tqdm import tqdm
    import torch
    import torch.nn as nn
    from torch_geometric.datasets import QM9
    from torch_geometric.loader import DataLoader
    import torch_geometric.transforms as T
    from torch_geometric.nn import NNConv
    import torch.nn.functional as F
    from DiffuseSampler import DiffusionModel
    from base_model import *
    import wandb
    import os
    from omegaconf import OmegaConf
    # from test_model import Encoder
    
    args_dict = arg_parse()
    config = OmegaConf.load(f'{args_dict["configs"]}.yaml')
    config = config.diffusion
    data = QM9(root='./practice_data', transform=None)

    """
    each batch is considered a hugh graph with many nodes and edges,
    in EGNN, they introduce the concept of l2 distance between nodes, 
    yet I am not including this (probably not) for now. 

    """
    
    """
    python mole_diffuse.py --configs configs
    """
    # api = wandb.Api()
    # run = api.run("dalenhsiao/Projects/pretrain_gnn/Runs/train_gnn_with_embedded_h_run1")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # configs
    wandb.init(
            project=config.experiment,
            name=config.experiment_run,
            config={
                str(key): value for key, value in config.items()
                })
    load_model_pth = os.path.join("models", f"{config.load_model_dir}.pth")
    model_pth = os.path.join("diffusion_models", f"{config.save_model_dir}.pth")

    # temperarily embedding
    dataloader = DataLoader(data, batch_size=config.batch_size, shuffle=True)
    embedding = nn.Embedding(5, 1).to(device)
    # GNN net
    # net = GNN(
    #     n_feat_in=5,
    #     layers=config.layers,
    #     time_emb_dim=4,
    #     fine_tune=args_dict["fine_tune"],
    #     freeze_pretrain=args_dict["freeze"]
    #     ).to(device)
    net = GNN(
        n_feat_in=5,
        layers=config.layers,
        latent_space_dims=config.latent_space_dims,
        time_emb_dim=4,
        fine_tune=config.fine_tune,
        freeze_pretrain=config.freeze
        ).to(device)
    
    net.load_state_dict(
        torch.load(load_model_pth),
        strict = False
        )
    # diffusion model
    diffusion = DiffusionModel(
        net,
        timesteps=config.diffuse_timesteps
    ).to(device)
    # training
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
    num_epochs = config.epochs
    timestep = config.diffuse_timesteps
    criterion = nn.MSELoss()
    best_loss = np.inf
    # weights = torch.tensor(
    #     [0.6507343348068023,
    #      0.9452815257785657,
    #      5.935209084917005,
    #      4.291072126883656,
    #      259.02613087395696]
    # ).to(device)
    # print(cat.shape[1])
    for epoch in range(num_epochs):
        running_loss = 0
        with tqdm(dataloader) as tepoch:
            for step, data in enumerate(tepoch):
                tepoch.set_description(f'Epoch {epoch}')
                # optimizer.zero_grad()
                data = data.to(device)
                x = data.x[:,:5].long().to(device)
                h = torch.flatten(embedding(x), start_dim= 1)
                ts = torch.randint(0, timestep, (data.x.shape[0],), device=device).long()
                h_noisy, noise = diffusion.sample_forward_diffuse_training(h, ts, device=device)
                pred_noise, h_0 = diffusion.model_prediction(h_noisy, ts, data.edge_index)
                loss = criterion(pred_noise, noise)
                loss.backward()
                optimizer.step()
                running_loss += (loss.item()/len(dataloader))
                metrics = {
                        "train/train_loss": loss.item(),
                        "global_step": epoch * len(dataloader) + step, 
                        }
                wandb.log(metrics)
                tepoch.set_postfix(train_loss = loss.item())
        wandb.log({"epoch": epoch, "train/epoch_train_loss": running_loss})
        if running_loss < best_loss:
            early_stop = 0
            print(f"model converges, {best_loss} -> {running_loss}")
            best_loss = running_loss
            print("saving best model ....")
            torch.save(net.state_dict(), model_pth)
            wandb.save(model_pth)
        else:
            early_stop += 1
            if early_stop == config.early_stopping:
                print(f"No improvement in {config.early_stopping}, finish training")
    wandb.finish()

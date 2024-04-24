# loading the pretrained model into diffusion
import argparse
def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", "-exp", type=str, default="gnn")
    parser.add_argument("--experiment_run", "-r", type=str, default="experiement_run")
    # parser.add_argument("--save_model_dir", type=str, default=os.path.realpath())
    parser.add_argument("--diffuse_timesteps", "-tsteps", type=int, default=100)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--early_stopping", "-es", type=int, default=5)
    parser.add_argument("--batch_size", "-bs", type=int, default=64)
    parser.add_argument("--layers", nargs='+', type=int, default=[32, 64, 128])
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-3)
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
    from based_model_4_4 import *
    import wandb 
    # from test_model import Encoder
    
    args_dict = arg_parse()
    data = QM9(root='./practice_data', transform=None)

    """
    each batch is considered a hugh graph with many nodes and edges,
    in EGNN, they introduce the concept of l2 distance between nodes, 
    yet I am not including this (probably not) for now. 

    """
    
    """
    python mole_diffuse.py --experiment diffusion_gnn --experiment_run diffusion_with_layer_norm  --max_epochs 20 --early_stopping 5 --batch_size 32 --layers 32 64 128 --learning_rate 1e-3
    """
    # api = wandb.Api()
    # run = api.run("dalenhsiao/Projects/pretrain_gnn/Runs/train_gnn_with_embedded_h_run1")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # configs
    wandb.init(
            project=args_dict["experiment"],
            name=args_dict["experiment_run"],
            config={
                "epochs": args_dict["max_epochs"],
                "early_stopping": args_dict["early_stopping"],
                "time_step": args_dict["diffuse_timesteps"],
                "batch_size": args_dict["batch_size"],
                "layers": args_dict["layers"],
                "lr": args_dict["learning_rate"]
                })
    config = wandb.config

    # temperarily embedding
    dataloader = DataLoader(data, batch_size=config.batch_size, shuffle=True)
    embedding = nn.Embedding(5, 1).to(device)
    # GNN net
    net = Net(
        n_feat_in=5,
        layers=config.layers,
        time_emb_dim=4
        ).to(device)
    net.load_state_dict(
        torch.load("./models/model_best_64_nn_embed.pth")
        )
    # diffusion model
    diffusion = DiffusionModel(
        net,
        timesteps=config.time_step
    ).to(device)
    # training
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
    num_epochs = config.epochs
    timestep = config.time_step
    criterion = nn.MSELoss()
    best_loss = np.inf
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
                # h = torch.concat((h, pos), axis=1)
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
            model_path = "./models/diffusion_model.pth"
            torch.save(net.state_dict(), model_path)
            wandb.save(model_path)
        else:
            early_stop += 1
            if early_stop == config.early_stopping:
                print(f"No improvement in {config.early_stopping}, finish training")

    wandb.finish()
            
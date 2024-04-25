from base_model import Net
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
import math
import numpy as np
import os.path
import argparse
def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", "-exp", type=str, default="gnn")
    parser.add_argument("--experiment_run", "-r", type=str, default="experiement_run")
    parser.add_argument("--save_model", type=str, default="model")
    
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--early_stopping", "-es", type=int, default=5)
    parser.add_argument("--batch_size", "-bs", type=int, default=64)
    parser.add_argument("--layers", nargs='+', type=int, default=[32, 64, 128])
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-3)
    args = parser.parse_args()
    args_dict = vars(args)
    return args_dict

"""
python pretrain_gnn.py --experiment pretrain_gnn --save_model model_NLL --experiment_run logsoftmax_in_the_last_layer  --max_epochs 20 --early_stopping 5 --batch_size 32 --layers 32 64 128 --learning_rate 1e-3
"""

if __name__ == "__main__":
    args_dict = arg_parse()
    # configs
    wandb.init(
            project=args_dict["experiment"],
            name=args_dict["experiment_run"],
            config={
                "epochs": args_dict["max_epochs"],
                "early_stopping": args_dict["early_stopping"],
                "batch_size": args_dict["batch_size"],
                "layers": args_dict["layers"],
                "lr": args_dict["learning_rate"]
                })
    config = wandb.config
    save_model_pf = os.path.join("models", f"{args_dict["save_model"]}.pth")
    data = QM9(root='./practice_data', transform=None)
    dataloader = DataLoader(data, batch_size=config.batch_size, shuffle=True)
    # dummy
    dummy = next(iter(dataloader))
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)
    # model
    net = Net(
        n_feat_in=5,
        layers=config.layers,
        time_emb_dim=4
        ).to(device)

# class imbalance weights
    weights = torch.tensor(
        [0.6507343348068023,
         0.9452815257785657,
         5.935209084917005,
         4.291072126883656,
         259.02613087395696]
    ).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
    # Softmax is already included in the loss function, hence, NLL is used instead
    criterion = nn.NLLLoss(weight = weights)
    n_steps_per_epoch = math.ceil(len(dataloader) / config.batch_size)
    best_loss = np.inf
    # import pdb ; pdb.set_trace()

    # embedding 
    embedding = nn.Embedding(5, 1).to(device)
    early_stop = 0
    for epoch in range(config.epochs):
        running_loss = 0
        with tqdm(dataloader) as tepoch:
            for step, data in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")
                optimizer.zero_grad()
                data = data.to(device)
                x = data.x[:, :5].to(device)
                h = torch.flatten(embedding(x.long()), start_dim=1)
                ts = torch.fill(torch.empty((data.x.shape[0], )), 0).to(device)
                out = net(h, ts, data.edge_index.to(device))
                # import pdb; pdb.set_trace()
                loss = criterion(out, torch.argmax(x, dim=1))
                running_loss += loss.item() / len(dataloader)
                loss.backward()
                optimizer.step()
                metrics = {
                        "train/train_loss": loss.item(),
                        "global_step": epoch * len(dataloader) + step, 
                        }
                wandb.log(metrics)
                tepoch.set_postfix(train_loss=loss.item())
        wandb.log({"epoch": epoch, "train/NLL_loss_epochs": running_loss})
        if running_loss < best_loss:
            early_stop = 0
            print(f"model converges, {best_loss} -> {running_loss}")
            best_loss = running_loss
            print("saving best model ....")
            torch.save(net.state_dict(), save_model_pf)
            wandb.save(save_model_pf)
        else:
            early_stop += 1
            if early_stop == config.early_stopping:
                print(f"No improvement in {config.early_stopping}, finish training")
    wandb.finish()

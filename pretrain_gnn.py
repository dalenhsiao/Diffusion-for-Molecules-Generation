from based_model_4_4 import Net
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
import math
import numpy as np


# configs
wandb.init(
        project="gnn-pretrain-batchsize-64",
        config={
            "epochs": 100,
            "early_stopping":5,
            "batch_size": 64,
            "layers": [32, 64, 128],
            "lr": 1e-3
            })
config = wandb.config

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
weights = torch.tensor([0.6507343348068023,
            0.9452815257785657,
            5.935209084917005,
            4.291072126883656,
            259.02613087395696]
)
optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
criterion = nn.CrossEntropyLoss()
n_steps_per_epoch = math.ceil(len(dataloader) / config.batch_size)

best_loss = np.inf


for epoch in range(config.epochs):
    running_loss = 0
    with tqdm(dataloader) as tepoch:
        for step, data in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch}")
            optimizer.zero_grad()
            data = data.to(device)
            x = data.x[:, :5]
            ts = torch.fill(torch.empty((data.x.shape[0], )), 0).to(device)
            out = net(x, ts, data.edge_index.to(device))
            loss = criterion(out, x)
            running_loss += loss.item() / len(dataloader)
            loss.backward()
            optimizer.step()
            metrics = {
                    "train/train_loss": loss.item(),
                    "train/epoch": (step + 1 + (n_steps_per_epoch * epoch)) / n_steps_per_epoch, 
                       }
            if step + 1 < n_steps_per_epoch:
                # Log train metrics to wandb
                wandb.log(metrics)
            tepoch.set_postfix(train_loss=loss.item())
    wandb.log({**metrics, "epoch": epoch, "train/epoch_train_loss": running_loss})
    if running_loss < best_loss:
        early_stop = 0
        print(f"model converges, {best_loss} -> {running_loss}")
        best_loss = running_loss
        print("saving best model ....")
        model_path = "./models/model_best_64.pth"
        torch.save(net.state_dict(), model_path)
        wandb.save(model_path)
    else:
        early_stop += 1
        if early_stop == config.early_stopping:
            print(f"No improvement in {config.early_stopping}, finish training")

wandb.finish()

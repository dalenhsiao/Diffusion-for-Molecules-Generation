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
        project="gnn-pretrain",
        config={
            "epochs": 10,
            "batch_size": 1,
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
    n_feat_in=dummy.x.shape[1],
    layers=config.layers,
    time_emb_dim=4
    ).to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
criterion = nn.CrossEntropyLoss()
n_steps_per_epoch = math.ceil(len(dataloader) / config.batch_size)

best_loss = np.inf()

for epoch in range(config.epochs):
    running_loss = 0
    with tqdm(dataloader) as tepoch:
        for step, data in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch}")
            optimizer.zero_grad()
            data = data.to(device)
            x = data.x[:, :5]
            ts = torch.fill(torch.empty((data.shapa[0], )), 0)
            out = net(x, ts, data.edge_index)
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
    wandb.log({**metrics, "epoch": epoch})
    if running_loss < best_loss:
        best_loss = running_loss
        print(f"model converges, {best_loss} -> {running_loss}")
        print("saving best model ....")
        model_path = "/models/model_best.pth"
        torch.save(net.state_dict(), model_path)
        wandb.save(model_path)

wandb.finish()

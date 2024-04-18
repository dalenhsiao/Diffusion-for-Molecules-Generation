# loading the pretrained model into diffusion

from DiffuseSampler import DiffuseSampler


# get loss for diffusion process
def get_loss(model, x_0, t, edge_index, total_timestep, device, mode="linear", edge_attr= None):
    """_summary_

    Args:
        x_0 (Tensor): ground truth data
        edge_index (Tensor): edge index -> shape = (2, edge)
        total_timestep (int): Total timesteps
        t (Tensor): timesteps sample -> (n_batch, )
        device : device
        mode (String): "linear", "cosine" schedule for noise scheduling 
        edge_attr (Tensor): edge attributes -> shape = (n_edge, n_edge_feat)

    Returns:
        _type_: _description_
    """
    # generate sample 
    
    x_noised, noise = DiffuseSampler.sample_forward_diffuse_training(x_0, total_timestep, t, device,mode) # noised, noise added
    pred_noise = model(x_noised, t, edge_index, edge_attr)
    metric = nn.MSELoss()
    loss = metric(pred_noise, noise)
    return loss

# def get_loss(pred_noise, x_0, t,total_timestep, device, mode="linear"):
#     """_summary_

#     Args:
#         x_0 (Tensor): ground truth data
#         edge_index (Tensor): edge index -> shape = (2, edge)
#         total_timestep (int): Total timesteps
#         t (Tensor): timesteps sample -> (n_batch, )
#         device : device
#         mode (String): "linear", "cosine" schedule for noise scheduling 
#         edge_attr (Tensor): edge attributes -> shape = (n_edge, n_edge_feat)

#     Returns:
#         _type_: _description_
#     """
#     # generate sample 
    
#     x_noised, noise = DiffuseSampler.sample_forward_diffuse_training(x_0, total_timestep, t, device,mode) # noised, noise added
#     print("x_noised", x_noised.shape)
#     print("noise ", noise.shape)
#     metric = nn.MSELoss()
#     loss = metric(pred_noise, noise)
#     return loss




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
    from based_model_4_4 import *
    # from test_model import Encoder
    data = QM9(root='./practice_data', transform=None)

    """
    each batch is considered a hugh graph with many nodes and edges,
    in EGNN, they introduce the concept of l2 distance between nodes, 
    yet I am not including this (probably not) for now. 


    """
    dataloader = DataLoader(data, batch_size=1, shuffle=True)
    # try the customized pyg model
    hidden_dim = 64  # hidden dimension
    n_feat_out = 7  # output latent embedding shape
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_epochs = 10
    timestep = 1000

    # temperarily embedding
    embedding = nn.Embedding(4, 4).to(device)
    encoder = Net(
        n_feat_in=5,
        layers=[32, 64, 128],
        time_emb_dim=4
        ).to(device)
    encoder.load_state_dict(
        torch.load("./models/model_best_64_es4.pth")
        )
    import pdb ; pdb.set_trace()
    # data parallel
    # if torch.cuda.device_count() > 1:
    #     print(f"Let's use {torch.cuda.device_count()} GPUs!")
    #     encoder = nn.parallel.DataParallel(encoder)
    encoder.to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)
    print(encoder)

    # print(cat.shape[1])
    for epoch in range(num_epochs):
        running_loss = 0
        with tqdm(dataloader) as tepoch:
            for data in tepoch:
                tepoch.set_description(f'Epoch {epoch}')
                # optimizer.zero_grad()
                data = data.to(device)
                # x = torch.concat((data.x[:,:4],data.pos), axis=1) # node features -> one hot vector + positional information
                x = data.x[:,:5].long().to(device)
                pos = data.pos.to(device)
                h = torch.flatten(embedding(x), start_dim= 1)
                ts = torch.randint(0, timestep, (data.x.shape[0],), device=device).long()
                h = torch.concat((h, pos), axis=1)
                # out = encoder(data.x, ts, data.edge_index)
                # pred_noise = encoder(h,ts, data.edge_index)
                loss = get_loss(encoder, h, ts, data.edge_index, timestep, device)
                # loss = get_loss(pred_noise, h,ts,timestep)
                loss.backward()
                optimizer.step()
                running_loss += (loss.item()/len(dataloader))
                tepoch.set_postfix(train_loss=running_loss)
            
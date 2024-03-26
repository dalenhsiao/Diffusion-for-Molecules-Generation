import torch 
import torch.nn as nn 
import math 
from torch_geometric.nn import NNConv, Set2Set
import torch.nn.functional as F




# # create message passing layer (Graph convolution)
# class GraphConv(nn.Module):
#     def __init__(self, node_inf, node_outf):
#         """Message passing 

#         Args:
#             node_inf (_type_): input node features (molecules charge, categorical types)
#             node_outf (_type_): output node features 
            
#             edge_inf (_type_): input edge info (bond distance(l2 norm))
            
#         """
#         super(GraphConv, self).__init__()
#         # a node mlp to learn node presentation
        
#         self.node_mlp = nn.Sequential(
#             nn.Linear(2, 16),
#             nn.ReLU(),
#             nn.Linear(16, 16)
#         )
#         # an edge mlp to learn edge presentation
#         self.edge_mlp = nn.Sequential(
#             nn.Linear(2, 16),
#             nn.ReLU(),
#             nn.Linear(16, 16)
#         )
        
#     def forward(self,h, adj_mat):
#         """_summary_

#         Args:
#             h (_type_): node features
#             adj_mat (_type_): _description_
#         """
#         # forward pass
#         self.node_mlp()
#         self.edge_mlp()
        
        
        
## pygeometrics

class Net(nn.Module):
    def __init__(self, n_feat_in, hidden_dim, latent_dim, n_layers, activation= nn.SiLU(),device="cpu"):
        super().__init__()
        self.device = device
        self.n_layers = n_layers
        self.act = activation
        self.embedding = nn.Linear(n_feat_in, hidden_dim) # initialize the embedding layer
        self.embedding_out = nn.Linear(hidden_dim, latent_dim) # output embedding layer (latent space)
        
        self.nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), 
            self.act,
            nn.Linear(hidden_dim, hidden_dim)
            ) # NN convolution block for edge conditioned convolution
        
        ### Encoder 
        # graph convolution layers
        for i in range(self.n_layers):
            # edge conditioned convolution (input is node features and conditioned by edge)
            self.add_module("gcl_%d" % i, 
                NNConv(in_channels=hidden_dim, out_channels=hidden_dim, nn =self.nn, aggr='add')
                ) # message passing layer
        self.to(self.device)
                1
        # Pooling layers
        """
        Basically the parameters
        
        graph convolution:
        - forward pass: (x:nodes embedding, edge_index: edge index, edge_attr: edge attributes)
        
        """
        self.pooling = Set2Set(latent_dim, processing_steps=4) # aggregation 
        
        # graph convolution layer
        # self.conv = NNConv(in_channels=hidden_dim, out_channels=hidden_dim, nn:callable, aggr='mean') # message passing layer 
        # self.gru = GRU(hidden_dim, hidden_dim) # gru layer
        
            
        
    
        # self.set2set = Set2Set(dim, processing_steps=3) 
        # self.lin1 = torch.nn.Linear(2 * dim, dim)
        # self.lin2 = torch.nn.Linear(dim, 1)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """_summary_

        Args:
            x (_type_): node features
            edge_index (_type_): edge index matrix, shape = [2, num_edges]
            edge_attr (_type_): edge attributes, default None

        Returns:
            h: graph latent representation
        """
        h = self.embedding(x)
        
        # out = F.relu(self.lin0(data.x))
        # h = out.unsqueeze(0)

        
        for i in range(self.n_layers): # num_layers
            # h = F.relu(self.conv(h, edge_index, edge_attr))
            h = self._modules["gcl_%d" % i](h, edge_index, edge_attr)
            # out, h = self.gru(m.unsqueeze(0), h)
            # out = out.squeeze(0)
        if batch is not None:
            out = self.set2set(out, data.batch) # pooling layer
        
        out = F.relu(self.embedding_out(h))
        
        # out = F.relu(self.lin1(out))
        # out = self.lin2(out)/
        return out


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Net().to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
#                                                        factor=0.7, patience=5,
#                                                        min_lr=0.00001)


# def train(epoch):
#     model.train()
#     loss_all = 0

#     for data in train_loader:
#         data = data.to(device)
#         optimizer.zero_grad()
#         loss = F.mse_loss(model(data), data.y)
#         loss.backward()
#         loss_all += loss.item() * data.num_graphs
#         optimizer.step()
#     return loss_all / len(train_loader.dataset)


# def test(loader):
#     model.eval()
#     error = 0

#     for data in loader:
#         data = data.to(device)
#         error += (model(data) * std - data.y * std).abs().sum().item()  # MAE
#     return error / len(loader.dataset)


# best_val_error = None
# for epoch in range(1, 301):
#     lr = scheduler.optimizer.param_groups[0]['lr']
#     loss = train(epoch)
#     val_error = test(val_loader)
#     scheduler.step(val_error)

#     if best_val_error is None or val_error <= best_val_error:
#         test_error = test(test_loader)
#         best_val_error = val_error

#     print(f'Epoch: {epoch:03d}, LR: {lr:7f}, Loss: {loss:.7f}, '
#           f'Val MAE: {val_error:.7f}, Test MAE: {test_error:.7f}')
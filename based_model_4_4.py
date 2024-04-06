import torch
import torch.nn as nn
import math
from torch_geometric.nn import NNConv, GATConv, GCNConv

# Graph Convolution Layer
class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, edge_attr_dim=0, aggr="mean",activation=nn.SiLU(), attention=False):
        super(GraphConv, self).__init__()
        self.activation = activation
        
        
        if edge_attr_dim > 0:
            nn_model = nn.Sequential(
                    nn.Linear(edge_attr_dim, 32), 
                    self.activation,
                    nn.Linear(32, in_channels*in_channels)
            )
            self.gc_edge_attr = NNConv(in_channels, in_channels, nn_model, aggr)
            self.linear = nn.Linear(in_channels, out_channels)
        
        
        self.time_mlp = nn.Linear(time_emb_dim, out_channels) # (n_sample, out_channels)
        

        
        if attention: 
            self.gc = GATConv(in_channels, out_channels)
        else:
            self.gc = GCNConv(in_channels, out_channels)
        
        self.lin = nn.Linear(out_channels, out_channels)
            
        
    
    # if edge_attr is None, will not convolve the edge attributes 
    def forward(self, x, edge_index,t, edge_attr=None):
        if edge_attr is None:
            x = self.activation(self.gc(x,edge_index))
        else:
            x = self.activation(self.gc_edge_attr(x, edge_index, edge_attr))
            x = self.activation(self.linear(x))
            
            
        time_emb = self.activation(self.time_mlp(t))
        # print("shape x: ", x.shape)
        # print("shape time_emb: ", time_emb.shape)
        x += time_emb
        x = self.activation(self.lin(x))
        
        return x

# for time embedding (Sinusoidal time embedding)
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings

# edge attributes NN
class NNEdgeAttr(nn.Module):
    def __init__(self, edge_in_dim, edge_out_dim, act):
        super(NNEdgeAttr, self).__init__()
        # if edge_attr:
        self.act = act 
        self.self.edge_attr_nn = nn.Sequential(
                nn.Linear(edge_in_dim, edge_out_dim), 
                self.act,
                nn.Linear(edge_out_dim, edge_out_dim*edge_out_dim)
                ) # NN convolution block for edge conditioned convolution
    def forward(self,x):
        out = self.edge_attr_nn(x)
        return out
        


# Encoder 
class Net(nn.Module):
    # def __init__(self, n_feat_in, hidden_dim, latent_dim, time_emb_dim,n_layers, activation= nn.SiLU(), edge_attr_dim = None):
    def __init__(self, n_feat_in, layers, time_emb_dim, activation= nn.SiLU(), edge_attr_dim = 0):
        super(Net, self).__init__()
        """
        n_feat_in: number of features input 
        layers: model structure 
        time_emb_dim: time embedding dimension 
        activation: activation function 
        edge_attr_dim: edge attributes dimensions (default 0 for not using edge attributes)
        """
        # self.device = device
        # self.n_layers = n_layers
        self.layers = layers
        self.act = activation
        self.embedding = nn.Linear(n_feat_in, self.layers[0]) # initialize the embedding layer
        self.embedding_out = nn.Linear(self.layers[0], n_feat_in) # output embedding layer (latent space)
        self.edge_attr_dim = edge_attr_dim
        # hidden_ = (64, 128, 256)
        
        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU() # maybe modify?
            )
        
    
        
        ### Encoder (Downsampling)
        # graph convolution layers
        
        self.downsampling = nn.ModuleList([
            GraphConv(in_channels=self.layers[i], out_channels=self.layers[i+1], time_emb_dim=time_emb_dim, edge_attr_dim = self.edge_attr_dim, aggr='add', activation=self.act)
            for i in range(0,len(self.layers)-1)
        ]
        )
        
        ### Decoder (Upsampling)
        self.upsampling = nn.ModuleList([
            GraphConv(in_channels=self.layers[i], out_channels=self.layers[i-1],time_emb_dim=time_emb_dim, edge_attr_dim=self.edge_attr_dim, aggr='add', activation=self.act)
            for i in reversed(range(1, len(self.layers)))
        ])
        
            
        # Pooling layers
        """
        Basically the parameters
        
        graph convolution:
        - forward pass: (x:nodes embedding, edge_index: edge index, edge_attr: edge attributes)
        
        """


    def forward(self, x, timestep,edge_index, edge_attr=None):
        """_summary_

        Args:
            x (_type_): node features
            edge_index (_type_): edge index matrix, shape = [2, num_edges]
            edge_attr (_type_): edge attributes, default None


        Returns:
            h: graph latent representation
        """
        h = self.embedding(x) # initialize node embedding
        t = self.time_mlp(timestep)
        # print("initial h:", h.shape)
        
        # for i in range(self.n_layers):
        #     # print("running gcl_%d" % i)
        #     # h = self._modules["gcl_%d" % i](h, edge_index, edge_attr)
        #     # print("h in the loop:", h.shape)
        #     h = self.
        
        # downsampling 
        for down in self.downsampling:
            h = down(h, edge_index, t, edge_attr)
        
        
        # bottle neck 
        
        
        # upsampling 
        for up in self.upsampling:
            h = up(h, edge_index, t, edge_attr)
            
        
        out = self.embedding_out(h)
    
        return out
    
    
    # get edge attributes NN 
    def getEdgeAttrNN(self, edge_in_dim, edge_out_dim):
        
        return nn.Sequential(
                nn.Linear(edge_in_dim, edge_out_dim), 
                self.act,
                nn.Linear(edge_out_dim, edge_out_dim*edge_out_dim)
        )
        
        

    

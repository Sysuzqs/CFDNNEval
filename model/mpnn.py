import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, List
from torch_geometric.nn import MessagePassing, InstanceNorm
from torch_geometric.data import Data
from torch_cluster import radius_graph


class Swish(nn.Module):
    """Swish activation function
    """
    def __init__(self, beta=1):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta*x)
    

class GNN_Layer(MessagePassing):
    """Message passing layer
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 hidden_features: int,
                 time_window: int,
                 spatial_dim: int,
                 n_variables: int):
        """Initialize message passing layers
        Args:
            in_features (int): number of node input features
            out_features (int): number of node output features
            hidden_features (int): number of hidden features
            time_window (int): number of input/output timesteps (temporal bundling)
            spatial_dim (int): number of dimension of spatial domain  
            n_variables (int): number of equation specific parameters used in the solver
        """
        super(GNN_Layer, self).__init__(node_dim=-2, aggr='mean') # node_dim: The axis along which to propagate. (default: -2)
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        assert (spatial_dim == 1 or spatial_dim == 2 or spatial_dim == 3)

        self.message_net_1 = nn.Sequential(nn.Linear(2 * in_features + time_window + spatial_dim + n_variables, hidden_features), 
                                           Swish()
                                           )
        self.message_net_2 = nn.Sequential(nn.Linear(hidden_features, hidden_features), 
                                           Swish()
                                           )
        self.update_net_1 = nn.Sequential(nn.Linear(in_features + hidden_features + n_variables, hidden_features), 
                                          Swish()
                                          )
        self.update_net_2 = nn.Sequential(nn.Linear(hidden_features, out_features), 
                                          Swish()
                                          )
        self.norm = InstanceNorm(hidden_features)

    def forward(self, x, u, pos, variables, edge_index, batch):
        """Propagate messages along edges
        """
        x = self.propagate(edge_index, x=x, u=u, pos=pos, variables=variables)
        x = self.norm(x, batch)
        return x

    def message(self, x_i, x_j, u_i, u_j, pos_i, pos_j, variables_i):
        """Message update following formula 8 of the paper
        """
        message = self.message_net_1(torch.cat((x_i, x_j, u_i - u_j, pos_i - pos_j, variables_i), dim=-1))
        message = self.message_net_2(message)
        return message

    def update(self, message, x, variables):
        """Node update following formula 9 of the paper
        """
        update = self.update_net_1(torch.cat((x, message, variables), dim=-1))
        update = self.update_net_2(update)
        if self.in_features == self.out_features:
            return x + update
        else:
            return update

        

class MPNN(nn.Module):
    def __init__(self, 
                 loss_fn: nn.Module,
                 neighbors: int = 1,
                 delta_t: float = 0.1,
                 hidden_features: int = 128,
                 hidden_layers: int = 6,
                 n_params: int = 5,
                 physics: str = "u"):
        super().__init__(loss_fn)
        self.loss_fn = loss_fn
        self.k = neighbors
        self.dt = delta_t
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.n_params = n_params

        assert physics in ("u", "v", "p")
        if physics == "u":
            self.v = 0
        elif physics == "v":
            self.v = 1
        else:
            self.v = 2

        # encoder
        self.embedding_mlp = nn.Sequential(
            nn.Linear(1+2+n_params, self.hidden_features), # f([u, x, y, parmas])
            Swish(),
            nn.Linear(self.hidden_features, self.hidden_features),
            Swish()
            )

        # processor
        self.gnn_layers = torch.nn.ModuleList(modules=(GNN_Layer(
            in_features=self.hidden_features,
            hidden_features=self.hidden_features,
            out_features=self.hidden_features,
            time_window=1,
            spatial_dim=2,
            n_variables=n_params
            ) for _ in range(self.hidden_layers)))
        
        # decoder
        self.output_mlp = nn.Sequential(
            nn.Linear(self.hidden_features, self.hidden_features // 2),
            Swish(),
            nn.Linear(self.hidden_features // 2, 1)
            )

    def forward(
        self,
        inputs: Tensor,
        label: Optional[Tensor] = None,
        case_params: Optional[dict] = None,
        mask: Optional[Tensor] = None,
        **kwargs,
    ) -> dict:
        device = inputs.device
        b, c, h, w = inputs.shape
        
        # pre-process data
        graph = self.create_graph(inputs, label, case_params, mask).to(device)
        u = graph.x[:, self.v].unsqueeze(-1) # [bs*nx, 1]
        x_pos = graph.pos # [bs*nx, 2]
        edge_index = graph.edge_index # [2, num_edges]
        batch = graph.batch # [bs*nx]
        params = graph.params # [bs*nx, num_params]

        # encode
        node_input = torch.cat([u, x_pos, params], dim=-1)
        f = self.embedding_mlp(node_input) # [bs*nx, hidden_dim]
        # process
        for i in range(self.hidden_layers):
            f = self.gnn_layers[i](f, u, x_pos, params, edge_index, batch) # [bs*nx, hidden_dim]
        # decode
        diff = self.output_mlp(f) # [bs*nx, 1]
        out = u + self.dt * diff # [bs*nx, 1]

        # post-process
        # loss = self.loss_fn(preds=out, labels=target)
        preds = torch.reshape(out, [b, h, w, 1]).permute([0, 3, 1, 2]) # [b, 1, h, w]
        if label is not None:
            labels = label[:, self.v, :, :].unsqueeze(1)
            loss = self.loss_fn(preds=preds, labels=labels)
            return dict(
                preds=preds,
                loss=loss,
            )

        return dict(preds=preds)
    
    def create_graph(self, 
                     inputs, 
                     label, 
                     case_params, 
                     mask):
        device = inputs.device

        b, c, h, w = inputs.shape
        inputs = inputs.permute([0, 2, 3, 1]).reshape([-1, c]) # [b, c, h, w] -> [b, h, w, c] -> [b*h*w, c]
        label = label.permute([0, 2, 3, 1]).reshape([-1, c]) # [b, c, h, w] -> [b, h, w, c] -> [b*h*w, c]
        batch = torch.arange(b).unsqueeze(-1).repeat(1, h*w).flatten().long() # [b*h*w]

        pos = torch.Tensor().to(device) # [b*h*w, 2]
        for i in range(b):
            x = torch.linspace(0, case_params[i, 1], w)
            y = torch.linspace(0, case_params[i, 0], h)
            gird_x, grid_y = torch.meshgrid(x, y, indexing='ij')
            xy = torch.stack([gird_x, grid_y], dim=-1).reshape([-1, 2]).to(device) # [h*w, 2]
            pos = torch.cat([pos, xy], dim=0)

        # edge index, x and y below is logical coordinate
        x = torch.arange(w).float()
        y = torch.arange(h).float()
        gird_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        xy = torch.stack([gird_x, grid_y], dim=-1).reshape([-1, 2]).repeat(b, 1) # [b*h*w, 2]
        edge_index = radius_graph(xy, r=np.sqrt(2)*self.k, batch=batch, loop=False)

        graph = Data(x=inputs, edge_index=edge_index)
        graph.y = label
        graph.pos = pos
        graph.batch = batch
        graph.params = case_params.repeat(1, h*w).reshape([b*h*w, -1]) # [b, num_params] -> [b*h*w, num_params]
        graph.validate(raise_on_error=True)
    
        return graph
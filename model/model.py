# model.py

import torch.nn as nn
import torch

class MLP(nn.Module):
    """
        Multi-Layers Perceptron Model
        Parameters:
            - num_features : int : the number of features (input nodes)
            - num_hidden_nodes : list or int : the number of hidden layer nodes
            - num_classes : int : the number of classes (output nodes)
            - activation : str : the name of activation fucntion
    """
    def __init__(self, num_features:int=2, num_hidden_nodes:list=[10], num_classes:int=2, activation:str='sigmoid'):
        super(MLP, self).__init__()
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        if isinstance(num_hidden_nodes, int):
            self.hidden_dim = [num_hidden_nodes]
        elif isinstance(num_hidden_nodes, list):
            self.hidden_dim = [int(n) for n in num_hidden_nodes]
        else:
            raise TypeError("num_hidden_nodes should be positive integer or list of positive integers")
        self.input_dim = num_features
        self.output_dim = num_classes
        current_dim = self.input_dim
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'linear' or activation == 'identity':
            self.activation = nn.Identity()
        else:
            raise TypeError("activation does not supported, activation should be in [\'sigmoid\', \'linear\']")

        self.layers = nn.ModuleList()
        for hdim in self.hidden_dim:
            self.layers.append(nn.Linear(current_dim, hdim).to(self.device))
            current_dim = hdim
        self.layers.append(nn.Linear(current_dim, self.output_dim).to(self.device))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        out = self.activation(x)
        return out
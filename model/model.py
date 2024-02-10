import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GNN(torch.nn.Module):
  def __init__(self, n_features, n_classes, n_hidden=64, drop_rate=0.3):
    super(GNN, self).__init__()
    self.conv1 = GCNConv(n_features, n_hidden)
    self.conv2 = GCNConv(n_hidden, n_classes)
    self.drop_rate = drop_rate

  def reset_parameters(self):
    self.conv1.reset_parameters()
    self.conv2.reset_parameters()

  def forward(self, data):
    x, edge_index = data.x, data.edge_index
    x = F.relu(self.conv1(x, edge_index))
    x = F.dropout(x, p= self.drop_rate, training= self.training)
    x = F.relu(self.conv2(x, edge_index))
    x = F.log_softmax(x, dim=1)  
    return x
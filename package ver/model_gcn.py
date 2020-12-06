import os
import dgl
import torch
import collections
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GraphConv

from data_loader_multi import data_loader_multi 


class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, num_classes)

    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = torch.relu(h)
        h = self.conv2(g, h)
        return h


def graph_generator(path: str):
    edges_origin, edges_trans_src, edges_trans_dst, src_in_degree_norm, src_out_degree_norm, dst_in_degree_norm, dst_out_degree_norm, port_num_norm, time_len_norm, connect_num_norm, _, _ = data_loader_multi(path)

    # Create Graph
    g = dgl.DGLGraph()

    # Create Nodes and Edges
    g.add_nodes(len(edges_origin))
    g.add_edges(edges_trans_src, edges_trans_dst)

    # Assign Features
    g.ndata['src_in_degree'] = torch.tensor(src_in_degree_norm)
    g.ndata['src_out_degree'] = torch.tensor(src_out_degree_norm)
    g.ndata['dst_in_degree'] = torch.tensor(dst_in_degree_norm)
    g.ndata['dst_out_degree'] = torch.tensor(dst_out_degree_norm)
    g.ndata['port_num'] = torch.tensor(port_num_norm)
    g.ndata['time_len'] = torch.tensor(time_len_norm)
    g.ndata['connect_num'] = torch.tensor(connect_num_norm)

    g = dgl.add_self_loop(g)

    return g


def gcn_train(path):
    _, _, _, _, _, _, _, _, _, _, feature_matrix, label_list= data_loader_multi(path)

    net = GCN(7, 16, 25)

    inputs = torch.tensor(feature_matrix)
    labels = torch.tensor(list(label_list.values()), dtype=torch.float32)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
    loss_func = nn.MSELoss()

    # Recoder
    loss_list = []
    accu_list = []

    for epoch in range(500):
        outputs = net(g, inputs)
        
        total_number = 0
        correct_number = 0
        
        outputs_max = np.argmax(outputs.detach().numpy(), axis=1)
        labels_max = np.argmax(labels.detach().numpy(), axis=1)
        
        for i in range(len(outputs_max)):
            if outputs_max[i] == labels_max[i]:
                correct_number += 1
            total_number += 1
        
        loss = loss_func(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_list.append(loss.item()/g.number_of_nodes())
        accu_list.append(correct_number / total_number)
        
        if epoch % 100 == 0:
            print('Epoch {} | Loss: {:.4f} | Accu: {:.4f}'.format(epoch, loss.item()/g.number_of_nodes(), correct_number / total_number))


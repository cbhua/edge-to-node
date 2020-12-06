import os
import dgl
import torch
import collections
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def label(flag: str) -> int:
    return {
        '-': 0,
        'snmpgetattack': 1,
        'warez': 2,
        'portsweep': 3,
        'nmap': 4,
        'warezclient': 5,
        'ipsweep': 6,
        'dict': 7,
        'neptune': 8,
        'smurf': 9,
        'pod': 10,
        'snmpguess': 11,
        'teardrop': 12,
        'satan': 13,
        'httptunnel-e': 14,
        'ignore': 15,
        'mscan': 16,
        'guest': 17,
        'rootkit': 18,
        'back': 19,
        'apache2': 20,
        'processtable': 21,
        'mailbomb': 22,
        'smurfttl': 23,
        'saint': 24
    }.get(flag[:-1])


def line_split(line: str) -> tuple:
    src_ip, dst_ip, port, time, connect_type = line.split('\t')
    return int(src_ip), int(dst_ip), int(port), int(time), label(connect_type)


def data_loader_multi(path: str) -> tuple:
    # Data File Path
    file_list = os.listdir(path)
    file_index = 0

    # Features
    src_in_degree = collections.OrderedDict()
    src_out_degree = collections.OrderedDict()
    dst_in_degree = collections.OrderedDict()
    dst_out_degree = collections.OrderedDict()

    port_num = collections.OrderedDict()
    port_list = collections.OrderedDict()
    time_len = collections.OrderedDict()
    connect_num = collections.OrderedDict()
    label_list = collections.OrderedDict()

    # Tool Parameters
    nodes_origin = []
    edges_origin = {}

    edges_trans_src = []
    edges_trans_dst = []

    node_in_degree = {}
    node_out_degree = {}

    node_in_edges = {}
    node_out_edges = {}

    edge_src_list = {}
    edge_dst_list = {}

    # Count Parameters
    edge_index = 0

    print('Init variables done.')

    for file in file_list:
        file_index += 1
        if file_index == 114:
            continue
        print('Loading File {}/{}...'.format(file_index, len(file_list)), end='\r')
        
        file_path = path + '/' + file
        data = open(file_path)
        
        for line in data:
            src_ip, dst_ip, port, time, connect_type = line_split(line)

            # Repeat Connections
            if (src_ip, dst_ip) in edges_origin.keys():
                edge_index_temp = edges_origin[(src_ip, dst_ip)]

                # Assign Features
                time_len[edge_index_temp] += time
                connect_num[edge_index_temp] += 1
                label_list[edge_index_temp][connect_type] += 1
                if port in port_list[edge_index_temp]:
                    pass
                else:
                    port_num[edge_index_temp] += 1
                    port_list[edge_index_temp].append(port)

            # New Connections     
            else:
                # Record Origin Node and Edge
                nodes_origin.append(src_ip)
                nodes_origin.append(dst_ip)
                edges_origin[(src_ip, dst_ip)] = edge_index

                # Update Node Parameters
                if src_ip in node_out_degree.keys():
                    node_out_degree[src_ip] += 1
                else:
                    node_out_degree[src_ip] = 1
                if dst_ip in node_in_degree.keys():
                    node_in_degree[dst_ip] += 1
                else:
                    node_in_degree[dst_ip] = 1

                # Store Nodes' In-Out Edges
                if src_ip in node_out_edges.keys():
                    node_out_edges[src_ip].append(edge_index)
                else:
                    node_out_edges[src_ip] = [edge_index]
                if dst_ip in node_in_edges.keys():
                    node_in_edges[dst_ip].append(edge_index)
                else:
                    node_in_edges[dst_ip] = [edge_index]

                # Update Edge Parameters
                if edge_index in edge_src_list.keys():
                    edge_src_list[edge_index].append(src_ip)
                else:
                    edge_src_list[edge_index] = [src_ip]
                if edge_index in edge_dst_list.keys():
                    edge_dst_list[edge_index].append(dst_ip)
                else:
                    edge_dst_list[edge_index] = [dst_ip]

                # Assign Features
                port_num[edge_index] = 1
                port_list[edge_index] = [port]
                time_len[edge_index] = time
                connect_num[edge_index] = 1
                label_list[edge_index] = [0 * x for x in range(25)]
                label_list[edge_index][connect_type] += 1

                edge_index += 1
                
    print('Data Loading Done')

    # Remove Repeats
    nodes_origin = list(set(nodes_origin))
    for i in range(len(edge_src_list)):
        edge_src_list[i] = list(set(edge_src_list[i]))
    for i in range(len(edge_dst_list)):
        edge_dst_list[i] = list(set(edge_dst_list[i]))
        
    # Create Transform Graph
    for node in nodes_origin:
        if not node in node_in_edges.keys() or not node in node_out_edges.keys():
            continue
        if not node_in_edges[node] or not node_out_edges[node]:
            continue
        for src_edge in node_in_edges[node]:
            for dst_edge in node_out_edges[node]:
                edges_trans_src.append(src_edge)
                edges_trans_dst.append(dst_edge)
                
    print('Data clean up done.')
        
    # Create Features
    for edge in edge_src_list.keys():
        if edge_src_list[edge][0] not in node_in_degree.keys():
            src_in_degree[edge] = 0
        else:
            src_in_degree[edge] = node_in_degree[edge_src_list[edge][0]]
        if edge_src_list[edge][0] not in node_out_degree.keys():
            src_out_degree[edge] = 0
        else:
            src_out_degree[edge] = node_out_degree[edge_src_list[edge][0]]
        
    for edge in edge_dst_list.keys():
        if edge_dst_list[edge][0] not in node_in_degree.keys():
            dst_in_degree[edge] = 0
        else:
            dst_in_degree[edge] = node_in_degree[edge_dst_list[edge][0]]
        if edge_dst_list[edge][0] not in node_out_degree.keys():
            dst_out_degree[edge] = 0
        else:
            dst_out_degree[edge] = node_out_degree[edge_dst_list[edge][0]]
        
    print('Feature create done.')
        
    src_in_degree_norm = list(src_in_degree.values()) / np.linalg.norm(list(src_in_degree.values()))
    src_out_degree_norm = list(src_out_degree.values()) / np.linalg.norm(list(src_out_degree.values()))
    dst_in_degree_norm = list(dst_in_degree.values()) / np.linalg.norm(list(dst_in_degree.values()))
    dst_out_degree_norm = list(dst_out_degree.values()) / np.linalg.norm(list(dst_out_degree.values()))

    port_num_norm = list(port_num.values()) / np.linalg.norm(list(port_num.values()))
    time_len_norm = list(time_len.values()) / np.linalg.norm(list(time_len.values()))
    connect_num_norm = list(connect_num.values()) / np.linalg.norm(list(connect_num.values()))

    print('Normalize done.')

    feature_matrix = np.empty([1, 7])
    for i in range(len(edges_origin)):
        feature_matrix = np.vstack((feature_matrix, [src_in_degree_norm[i], src_out_degree_norm[i], dst_in_degree_norm[i], dst_out_degree_norm[i], \
                                                    port_num_norm[i], time_len_norm[i], connect_num_norm[i]]))
    feature_matrix = np.delete(feature_matrix, obj=0, axis=0)    
    feature_matrix = feature_matrix.astype(np.float32)

    print('Data process done.')

    return (
        edges_origin, 
        edges_trans_src, 
        edges_trans_dst,
        src_in_degree_norm, 
        src_out_degree_norm, 
        dst_in_degree_norm, 
        dst_out_degree_norm, 
        port_num_norm, 
        time_len_norm, 
        connect_num_norm, 
        feature_matrix, 
        label_list
    )

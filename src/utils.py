import os
import dgl
import torch
import shutil
import random
import numpy as np


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    dgl.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_writable(path, overwrite=True):
    if not os.path.exists(path):
        os.makedirs(path)
    elif overwrite:
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        pass
        

# Calculation of evaluation metrics
def evaluat_metrics(output, label):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    pre_y = (torch.sigmoid(output) > 0.5).numpy()
    truth_y = label.numpy()
    N, C = pre_y.shape

    for i in range(N):
        for j in range(C):
            if pre_y[i][j] == truth_y[i][j]:
                if truth_y[i][j] == 1:
                    TP += 1
                else:
                    TN += 1
            elif truth_y[i][j] == 1:
                FN += 1
            elif truth_y[i][j] == 0:
                FP += 1

        # Accuracy = (TP + TN) / (N*C + 1e-10)
        Precision = TP / (TP + FP + 1e-10)
        Recall = TP / (TP + FN + 1e-10)
        F1_score = 2 * Precision * Recall / (Precision + Recall + 1e-10)

    return F1_score


# Data splitting by BFS
def get_bfs_sub_graph(ppi_list, node_num, node_to_edge_index, sub_graph_size):
    candiate_node = []
    selected_edge_index = []
    selected_node = []

    random_node = random.randint(0, node_num - 1)
    while len(node_to_edge_index[random_node]) > 20:
        random_node = random.randint(0, node_num - 1)
    candiate_node.append(random_node)

    while len(selected_edge_index) < sub_graph_size:
        cur_node = candiate_node.pop(0)
        selected_node.append(cur_node)

        for edge_index in node_to_edge_index[cur_node]:
            if edge_index not in selected_edge_index:
                selected_edge_index.append(edge_index)

                end_node = -1
                if ppi_list[edge_index][0] == cur_node:
                    end_node = ppi_list[edge_index][1]
                else:
                    end_node = ppi_list[edge_index][0]

                if end_node not in selected_node and end_node not in candiate_node:
                    candiate_node.append(end_node)
            else:
                continue

    # node_list = candiate_node + selected_node

    return selected_edge_index


# Data splitting by DFS
def get_dfs_sub_graph(ppi_list, node_num, node_to_edge_index, sub_graph_size):
    stack = []
    selected_edge_index = []
    selected_node = []

    random_node = random.randint(0, node_num - 1)
    while len(node_to_edge_index[random_node]) > 20:
        random_node = random.randint(0, node_num - 1)
    stack.append(random_node)

    while len(selected_edge_index) < sub_graph_size:
        cur_node = stack[-1]

        if cur_node in selected_node:
            flag = True

            for edge_index in node_to_edge_index[cur_node]:
                if flag:
                    end_node = -1

                    if ppi_list[edge_index][0] == cur_node:
                        end_node = ppi_list[edge_index][1]
                    else:
                        end_node = ppi_list[edge_index][0]

                    if end_node in selected_node:
                        continue
                    else:
                        stack.append(end_node)
                        flag = False
                else:
                    break

            if flag:
                stack.pop()
            continue

        else:
            selected_node.append(cur_node)

            for edge_index in node_to_edge_index[cur_node]:
                if edge_index not in selected_edge_index:
                    selected_edge_index.append(edge_index)

    return selected_edge_index
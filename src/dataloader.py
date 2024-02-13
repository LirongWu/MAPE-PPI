import os
import csv
import json
import pickle
import random
import numpy as np
from tqdm import tqdm

import dgl 
import torch

from utils import get_bfs_sub_graph, get_dfs_sub_graph

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data(dataset, split_mode, seed, skip_head=True):

    name = 0
    ppi_name = 0
    
    protein_name = {}
    ppi_dict = {}
    ppi_list = []
    ppi_label_list = []

    class_map = {'reaction':0, 'binding':1, 'ptmod':2, 'activation':3, 'inhibition':4, 'catalysis':5, 'expression':6}

    ppi_path = '../data/processed_data/protein.actions.{}.txt'.format(dataset)
    prot_seq_path = '../data/processed_data/protein.{}.sequences.dictionary.csv'.format(dataset)
    prot_r_edge_path = '../data/processed_data/protein.rball.edges.{}.npy'.format(dataset)
    prot_k_edge_path = '../data/processed_data/protein.knn.edges.{}.npy'.format(dataset)
    prot_node_path = '../data/processed_data/protein.nodes.{}.pt'.format(dataset)

    if os.path.exists("../data/processed_data/{}_ppi.pkl".format(dataset)):
        with open("../data/processed_data/{}_ppi.pkl".format(dataset), "rb") as tf:
            ppi_list = pickle.load(tf)
        with open("../data/processed_data/{}_ppi_label.pkl".format(dataset), "rb") as tf:
            ppi_label_list = pickle.load(tf)

    else:
        
        # get node and node name
        with open(prot_seq_path) as f:
            reader = csv.reader(f)
            for row in reader:
                protein_name[row[0]] = name
                name += 1

        for line in tqdm(open(ppi_path)):
            if skip_head:
                skip_head = False
                continue
            line = line.strip().split('\t')
        
            # if line[0] not in protein_name.keys():
            #     protein_name[line[0]] = name
            #     name += 1
            
            # if line[1] not in protein_name.keys():
            #     protein_name[line[1]] = name
            #     name += 1

            # get edge and its label
            if line[0] < line[1]:
                temp_data = line[0] + "__" + line[1]
            else:
                temp_data = line[1] + "__" + line[0]

            if temp_data not in ppi_dict.keys():
                ppi_dict[temp_data] = ppi_name
                temp_label = [0, 0, 0, 0, 0, 0, 0]
                temp_label[class_map[line[2]]] = 1
                ppi_label_list.append(temp_label)
                ppi_name += 1
            else:
                index = ppi_dict[temp_data]
                temp_label = ppi_label_list[index]
                temp_label[class_map[line[2]]] = 1
                ppi_label_list[index] = temp_label

        for ppi in tqdm(ppi_dict.keys()):
            temp = ppi.strip().split('__')
            ppi_list.append(temp)

        ppi_num = len(ppi_list)
        for i in tqdm(range(ppi_num)):
            seq1_name = ppi_list[i][0]
            seq2_name = ppi_list[i][1]
            ppi_list[i][0] = protein_name[seq1_name]
            ppi_list[i][1] = protein_name[seq2_name]

        with open("../data/processed_data/{}_ppi.pkl".format(dataset), "wb") as tf:
            pickle.dump(ppi_list, tf)
        with open("../data/processed_data/{}_ppi_label.pkl".format(dataset), "wb") as tf:
            pickle.dump(ppi_label_list, tf)

    ppi_g = dgl.to_bidirected(dgl.graph(ppi_list))
    protein_data = ProteinDatasetDGL(prot_r_edge_path, prot_k_edge_path, prot_node_path, dataset)
    ppi_split_dict = split_dataset(ppi_list, dataset, split_mode, seed)

    return protein_data, ppi_g.to(device), ppi_list, torch.FloatTensor(np.array(ppi_label_list)).to(device), ppi_split_dict


class ProteinDatasetDGL(torch.utils.data.Dataset):
    def __init__(self, prot_r_edge_path, prot_k_edge_path, prot_node_path, dataset):
        
        if os.path.exists("../data/processed_data/{}_protein_graphs.pkl".format(dataset)):
            with open("../data/processed_data/{}_protein_graphs.pkl".format(dataset), "rb") as tf:
                self.prot_graph_list = pickle.load(tf)
        
        else:

            prot_r_edge = np.load(prot_r_edge_path, allow_pickle=True)
            prot_k_edge = np.load(prot_k_edge_path, allow_pickle=True)
            prot_node = torch.load(prot_node_path)

            self.prot_graph_list = []

            for i in range(len(prot_r_edge)):
                prot_seq = []
                for j in range(prot_node[i].shape[0]-1):
                    prot_seq.append((j, j+1))
                    prot_seq.append((j+1, j))

                # prot_g = dgl.graph(prot_edge[i]).to(device)
                prot_g = dgl.heterograph({('amino_acid', 'SEQ', 'amino_acid') : prot_seq, 
                                          ('amino_acid', 'STR_KNN', 'amino_acid') : prot_k_edge[i],
                                          ('amino_acid', 'STR_DIS', 'amino_acid') : prot_r_edge[i]}).to(device)
                prot_g.ndata['x'] = torch.FloatTensor(prot_node[i]).to(device)

                self.prot_graph_list.append(prot_g)

            with open("../data/processed_data/{}_protein_graphs.pkl".format(dataset), "wb") as tf:
                pickle.dump(self.prot_graph_list, tf)

    def __len__(self):
        return len(self.prot_graph_list)

    def __getitem__(self, idx):
        return self.prot_graph_list[idx]
        
def collate(samples):
    return dgl.batch_hetero(samples)


def split_dataset(ppi_list, dataset, split_mode, seed):
    if not os.path.exists("../data/processed_data/{}_{}.json".format(dataset, split_mode)):
        if split_mode == 'random':
            ppi_num = len(ppi_list)
            random_list = [i for i in range(ppi_num)]
            random.shuffle(random_list)

            ppi_split_dict = {}
            ppi_split_dict['train_index'] = random_list[: int(ppi_num*0.6)]
            ppi_split_dict['val_index'] = random_list[int(ppi_num*0.6) : int(ppi_num*0.8)]
            ppi_split_dict['test_index'] = random_list[int(ppi_num*0.8) :]

            jsobj = json.dumps(ppi_split_dict)
            with open("../data/processed_data/{}_{}.json".format(dataset, split_mode), 'w') as f:
                f.write(jsobj)
                f.close()

        elif split_mode == 'bfs' or split_mode == 'dfs':
            node_to_edge_index = {}
            ppi_num = len(ppi_list)

            for i in range(ppi_num):
                edge = ppi_list[i]
                if edge[0] not in node_to_edge_index.keys():
                    node_to_edge_index[edge[0]] = []
                node_to_edge_index[edge[0]].append(i)

                if edge[1] not in node_to_edge_index.keys():
                    node_to_edge_index[edge[1]] = []
                node_to_edge_index[edge[1]].append(i)
            
            node_num = len(node_to_edge_index)
            sub_graph_size = int(ppi_num * 0.4)

            if split_mode == 'bfs':
                selected_edge_index = get_bfs_sub_graph(ppi_list, node_num, node_to_edge_index, sub_graph_size)
            elif split_mode == 'dfs':
                selected_edge_index = get_dfs_sub_graph(ppi_list, node_num, node_to_edge_index, sub_graph_size)
            
            all_edge_index = [i for i in range(ppi_num)]
            unselected_edge_index = list(set(all_edge_index).difference(set(selected_edge_index)))

            random_list = [i for i in range(len(selected_edge_index))]
            random.shuffle(random_list)

            ppi_split_dict = {}
            ppi_split_dict['train_index'] = unselected_edge_index
            ppi_split_dict['val_index'] = [selected_edge_index[i] for i in random_list[:int(ppi_num*0.2)]]
            ppi_split_dict['test_index'] = [selected_edge_index[i] for i in random_list[int(ppi_num*0.2):]]

            jsobj = json.dumps(ppi_split_dict)
            with open("../data/processed_data/{}_{}.json".format(dataset, split_mode), 'w') as f:
                f.write(jsobj)
                f.close()
        
        else:
            print("your mode is {}, you should use bfs, dfs or random".format(split_mode))
            return
    else:
        with open("../data/processed_data/{}_{}.json".format(dataset, split_mode), 'r') as f:
            ppi_split_dict = json.load(f)
            f.close()

    print("Train_PPI: {} | Val_PPI: {} | Test_PPI: {}".format(len(ppi_split_dict['train_index']), len(ppi_split_dict['val_index']), len(ppi_split_dict['test_index'])))

    return ppi_split_dict
    

def load_pretrain_data(dataset):

    prot_r_edge_path = '../data/processed_data/protein.rball.edges.{}.npy'.format(dataset)
    prot_k_edge_path = '../data/processed_data/protein.knn.edges.{}.npy'.format(dataset)
    prot_node_path = '../data/processed_data/protein.nodes.{}.pt'.format(dataset)

    protein_data = ProteinDatasetDGL(prot_r_edge_path, prot_k_edge_path, prot_node_path, dataset)

    return protein_data
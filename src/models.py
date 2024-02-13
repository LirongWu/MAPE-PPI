import dgl
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GraphConv, GINConv, HeteroGraphConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GIN(torch.nn.Module):
    def __init__(self,  param):
        super(GIN, self).__init__()

        self.num_layers = param['ppi_num_layers']
        self.dropout = nn.Dropout(param['dropout_ratio'])
        self.layers = nn.ModuleList()
        
        self.layers.append(GINConv(nn.Sequential(nn.Linear(param['prot_hidden_dim'] * 2, param['ppi_hidden_dim']), 
                                                 nn.ReLU(), 
                                                 nn.Linear(param['ppi_hidden_dim'], param['ppi_hidden_dim']), 
                                                 nn.ReLU(), 
                                                 nn.BatchNorm1d(param['ppi_hidden_dim'])), 
                                                 aggregator_type='sum', 
                                                 learn_eps=True))

        for i in range(self.num_layers - 1):
            self.layers.append(GINConv(nn.Sequential(nn.Linear(param['ppi_hidden_dim'], param['ppi_hidden_dim']), 
                                                     nn.ReLU(), 
                                                     nn.Linear(param['ppi_hidden_dim'], param['ppi_hidden_dim']), 
                                                     nn.ReLU(), 
                                                     nn.BatchNorm1d(param['ppi_hidden_dim'])), 
                                                     aggregator_type='sum', 
                                                     learn_eps=True))

        self.linear = nn.Linear(param['ppi_hidden_dim'], param['ppi_hidden_dim'])
        self.fc = nn.Linear(param['ppi_hidden_dim'], param['output_dim'])

    def forward(self, g, x, ppi_list, idx):

        for l, layer in enumerate(self.layers):
            x = layer(g, x)
            x = self.dropout(x)

        x = F.dropout(F.relu(self.linear(x)), p=0.5, training=self.training)

        node_id = np.array(ppi_list)[idx]
        x1 = x[node_id[:, 0]]
        x2 = x[node_id[:, 1]]

        x = self.fc(torch.mul(x1, x2))
        
        return x


class GCN_Encoder(nn.Module):
    def __init__(self, param, data_loader):
        super(GCN_Encoder, self).__init__()
        
        self.data_loader = data_loader
        self.num_layers = param['prot_num_layers']
        self.dropout = nn.Dropout(param['dropout_ratio'])
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.fc = nn.ModuleList()

        self.norms.append(nn.BatchNorm1d(param['prot_hidden_dim']))
        self.fc.append(nn.Linear(param['prot_hidden_dim'], param['prot_hidden_dim']))
        self.layers.append(HeteroGraphConv({'SEQ' : GraphConv(param['input_dim'], param['prot_hidden_dim']), 
                                            'STR_KNN' : GraphConv(param['input_dim'], param['prot_hidden_dim']), 
                                            'STR_DIS' : GraphConv(param['input_dim'], param['prot_hidden_dim'])}, aggregate='sum'))

        for i in range(self.num_layers - 1):
            self.norms.append(nn.BatchNorm1d(param['prot_hidden_dim']))
            self.fc.append(nn.Linear(param['prot_hidden_dim'], param['prot_hidden_dim']))
            self.layers.append(HeteroGraphConv({'SEQ' : GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim']), 
                                                'STR_KNN' : GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim']), 
                                                'STR_DIS' : GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim'])}, aggregate='sum'))

    def forward(self, vq_layer):

        prot_embed_list = []

        for iter, batch_graph in enumerate(self.data_loader):

            batch_graph.to(device)
            h = self.encoding(batch_graph)
            z, _, _ = vq_layer(h)
            batch_graph.ndata['h'] = torch.cat([h, z], dim=-1)
            prot_embed = dgl.mean_nodes(batch_graph, 'h').detach().cpu()
            prot_embed_list.append(prot_embed)

        return torch.cat(prot_embed_list, dim=0)


    def encoding(self, batch_graph):

        x = batch_graph.ndata['x']

        for l, layer in enumerate(self.layers):
            x = layer(batch_graph, {'amino_acid': x})
            x = self.norms[l](F.relu(self.fc[l](x['amino_acid'])))
            if l != self.num_layers - 1:
                x = self.dropout(x)

        return x
        

class GCN_Decoder(nn.Module):
    def __init__(self, param):
        super(GCN_Decoder, self).__init__()
        
        self.num_layers = param['prot_num_layers']
        self.dropout = nn.Dropout(param['dropout_ratio'])
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.fc = nn.ModuleList()

        for i in range(self.num_layers - 1):
            self.norms.append(nn.BatchNorm1d(param['prot_hidden_dim']))
            self.fc.append(nn.Linear(param['prot_hidden_dim'], param['prot_hidden_dim']))
            self.layers.append(HeteroGraphConv({'SEQ' : GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim']), 
                                                'STR_KNN' : GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim']), 
                                                'STR_DIS' : GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim'])}, aggregate='sum'))

        self.fc.append(nn.Linear(param['prot_hidden_dim'], param['input_dim']))
        self.layers.append(HeteroGraphConv({'SEQ' : GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim']), 
                                            'STR_KNN' : GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim']), 
                                            'STR_DIS' : GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim'])}, aggregate='sum'))


    def decoding(self, batch_graph, x):

        for l, layer in enumerate(self.layers):
            x = layer(batch_graph, {'amino_acid': x})
            x = self.fc[l](x['amino_acid'])

            if l != self.num_layers - 1:
                x = self.dropout(self.norms[l](F.relu(x)))
            else:
                pass

        return x


class CodeBook(nn.Module):
    def __init__(self, param, data_loader):
        super(CodeBook, self).__init__()

        self.param = param

        self.Protein_Encoder = GCN_Encoder(param, data_loader)
        self.Protein_Decoder = GCN_Decoder(param)

        self.vq_layer = VectorQuantizer(param['prot_hidden_dim'], param['num_embeddings'], param['commitment_cost'])

    def forward(self, batch_graph):
        z = self.Protein_Encoder.encoding(batch_graph)
        e, e_q_loss, encoding_indices = self.vq_layer(z)

        x_recon = self.Protein_Decoder.decoding(batch_graph, e)
        recon_loss = F.mse_loss(x_recon, batch_graph.ndata['x'])

        mask = torch.bernoulli(torch.full(size=(self.param['num_embeddings'],), fill_value=self.param['mask_ratio'])).bool().to(device)
        mask_index = mask[encoding_indices]
        e[mask_index] = 0.0

        x_mask_recon = self.Protein_Decoder.decoding(batch_graph, e)


        x = F.normalize(x_mask_recon[mask_index], p=2, dim=-1, eps=1e-12)
        y = F.normalize(batch_graph.ndata['x'][mask_index], p=2, dim=-1, eps=1e-12)
        mask_loss = ((1 - (x * y).sum(dim=-1)).pow_(self.param['sce_scale']))
        
        return z, e, e_q_loss, recon_loss, mask_loss.sum() / (mask_loss.shape[0] + 1e-12)


class VectorQuantizer(nn.Module):
    """
    VQ-VAE layer: Input any tensor to be quantized. 
    Args:
        embedding_dim (int): the dimensionality of the tensors in the
        quantized space. Inputs to the modules must be in this format as well.
        num_embeddings (int): the number of vectors in the quantized space.
        commitment_cost (float): scalar which controls the weighting of the loss terms.
    """
    def __init__(self, embedding_dim, num_embeddings, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        # initialize embeddings
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        
    def forward(self, x):    
        x = F.normalize(x, p=2, dim=-1)
        encoding_indices = self.get_code_indices(x)
        quantized = self.quantize(encoding_indices)

        q_latent_loss = F.mse_loss(quantized, x.detach())
        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = x + (quantized - x).detach().contiguous()

        return quantized, loss, encoding_indices
    
    def get_code_indices(self, x):

        distances = (
            torch.sum(x ** 2, dim=-1, keepdim=True) +
            torch.sum(F.normalize(self.embeddings.weight, p=2, dim=-1) ** 2, dim=1) -
            2. * torch.matmul(x, F.normalize(self.embeddings.weight.t(), p=2, dim=0))
        )
        
        encoding_indices = torch.argmin(distances, dim=1)
        
        return encoding_indices
    
    def quantize(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        return F.normalize(self.embeddings(encoding_indices), p=2, dim=-1)
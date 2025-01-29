import os
import time

import numpy as np
import scipy.sparse as sp
import sklearn
import torch
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances,cosine_similarity
import torch.nn.functional as F
import torch.nn as nn

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_columns(mx):
    """Column-normalize sparse matrix"""
    colsum = np.array(mx.sum(0))
    c_inv = np.power(colsum, -1).flatten()
    c_inv[np.isinf(c_inv)] = 0.
    c_mat_inv = sp.diags(c_inv)
    mx = mx.dot(c_mat_inv)
    return mx

def normalize_rol_cow(mx):
    """Symmetrically normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))  # Sum of elements in each row (degree, including self-loops)
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()  # D^{-1/2}
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  # Construct diagonal matrix of D^{-1/2}

    # Return D^{-1/2} * A * D^{-1/2}
    return d_mat_inv_sqrt.dot(mx).dot(d_mat_inv_sqrt)


def load_data(config):
    f = np.loadtxt(config.feature_path.format(config.name), dtype=float)
    features = sp.csr_matrix(f, dtype=np.float32)
    features = normalize(features)
    features = torch.FloatTensor(np.array(features.todense()))

    return features


# def load_data(config):
#     # 读取数据
#     f = np.loadtxt(config.feature_path.format(config.name), dtype=float)
#
#     f = f.reshape(-1, 1)
#
#     # 转换为稀疏矩阵并归一化
#     features = sp.csr_matrix(f, dtype=np.float32)
#     features = normalize(features)
#
#     # 转换为 PyTorch 的 FloatTensor
#     features = torch.FloatTensor(np.array(features.todense()))
#
#     return features


def load_data_std(config):
    f = np.loadtxt(config.feature_path.format(config.name), dtype=float)
    features = sp.csr_matrix(f, dtype=np.float32)
    features = torch.FloatTensor(np.array(features.todense()))
    return features

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_graph(config):
    struct_edges = np.genfromtxt(config.structgraph_path.format(config.name), dtype=np.int32)
    sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)

    sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])), shape=(config.n, config.n),
                         dtype=np.float32)

    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)

    sadj_with_self_loops = sadj + sp.eye(sadj.shape[0])

    nsadj = normalize(sadj_with_self_loops)

    nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)

    sadj_with_self_loops_torch = sparse_mx_to_torch_sparse_tensor(sadj_with_self_loops)

    return nsadj, sadj_with_self_loops_torch


def load_graph_std(config):
    struct_edges = np.genfromtxt(config.structgraph_path.format(config.name), dtype=np.int32)
    sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)

    sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])), shape=(config.n, config.n),
                         dtype=np.float32)

    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)

    sadj_with_self_loops = sadj + sp.eye(sadj.shape[0])

    nsadj = normalize_rol_cow(sadj_with_self_loops)

    nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)

    return nsadj


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def get_metric_matrix(matrix1, matrix2, method):

    assert method in ['Euclid', 'cosine'], "Unkown operation!"  # Assert the reasonability of measurement method
    if method == 'cosine':
        metric_matrix = cosine_similarity(matrix1, matrix2)
    else:
        metric_matrix = euclidean_distances(matrix1, matrix2)
    if method == 'cosine':  # In the case of cosine, logarithmic operations are required to ensure that the lower
        # the value for the better performance
        metric_matrix = np.exp(-metric_matrix)
    return metric_matrix


from collections import Counter
def augement_nodes(s_a, t_a):
    s = torch.zeros(s_a.shape[0])
    t = torch.zeros(t_a.shape[0])
    d_S = np.nonzero(s_a).t()[0].cpu().numpy().tolist()
    d_T = np.nonzero(t_a).t()[0].cpu().numpy().tolist()
    dix_s = Counter(d_S)
    dix_t = Counter(d_T)
    list_dix_s = sorted(dict(dix_s).items(), key= lambda x:x[1], reverse=True)
    list_dix_t = sorted(dict(dix_t).items(), key= lambda x:x[1], reverse=True)
    tensor_dix_s = torch.from_numpy(np.array(list_dix_s))
    tensor_dix_t = torch.from_numpy(np.array(list_dix_t))

    s_10 = tensor_dix_s[:10, :]
    t_10 = tensor_dix_t[:10, :]
    s_degree = s_10.t()[0].tolist()
    t_degree = t_10.t()[0].tolist()

    return torch.LongTensor(s_degree), torch.LongTensor(t_degree)

def get_topological_all(address):
    s, t = np.load(os.path.join('./data', address, address + '_s_centrality.npy'), allow_pickle=True), np.load(os.path.join('./data', address, address + '_t_centrality.npy'), allow_pickle=True)
    
    return s, t

if __name__ == '__main__':
    pass
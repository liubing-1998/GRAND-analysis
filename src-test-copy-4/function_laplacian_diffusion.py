import sys

import torch
from torch import nn
import torch_sparse

from base_classes import ODEFunc
from utils import MaxNFEException

import numpy as np
from scipy.sparse import coo_matrix

# Define the ODE function.
# Input:
# --- t: A tensor with shape [], meaning the current time.
# --- x: A tensor with shape [#batches, dims], meaning the value of x at t.
# Output:
# --- dx/dt: A tensor with shape [#batches, dims], meaning the derivative of x at t.
class LaplacianODEFunc(ODEFunc):

  # currently requires in_features = out_features
  def __init__(self, in_features, out_features, opt, data, device):
    super(LaplacianODEFunc, self).__init__(opt, data, device)

    self.in_features = in_features
    self.out_features = out_features
    self.w = nn.Parameter(torch.eye(opt['hidden_dim']))  # 80*80
    self.d = nn.Parameter(torch.zeros(opt['hidden_dim']) + 1)  # 80个1
    self.alpha_sc = nn.Parameter(torch.ones(1))
    self.beta_sc = nn.Parameter(torch.ones(1))


  def sparse_multiply(self, x):
    if self.opt['block'] in ['attention']:  # adj is a multihead attention
      mean_attention = self.attention_weights.mean(dim=1)
      ax = torch_sparse.spmm(self.edge_index, mean_attention, x.shape[0], x.shape[0], x)
    elif self.opt['block'] in ['mixed', 'hard_attention']:  # adj is a torch sparse matrix
      ax = torch_sparse.spmm(self.edge_index, self.attention_weights, x.shape[0], x.shape[0], x)
    else:  # adj is a torch sparse matrix
      ax = torch_sparse.spmm(self.edge_index, self.edge_weight, x.shape[0], x.shape[0], x)
    return ax

  def weight_diff(self, x):

    # 获取节点数量
    n = max(max(self.edge_index[0]), max(self.edge_index[1])) + 1
    # 创建稀疏矩阵
    data = np.ones(len(self.edge_index[0]))
    row_indices, col_indices = self.edge_index.cpu()
    sparse_adj_matrix = coo_matrix((data, (row_indices, col_indices)), shape=(n, n))

    # 计算每个节点与其相邻节点特征向量的加权差并归一化权重
    x = x.cpu().detach().numpy()
    weighted_differences = np.zeros_like(x)
    for i in range(sparse_adj_matrix.shape[0]):   # A*AT???
      neighbors = sparse_adj_matrix.getrow(i).indices  # 找出相邻节点的索引
      if len(neighbors) > 0:
        diffs = x[neighbors] - x[i]
        distances = np.linalg.norm(diffs, axis=1)  # 计算差值的范数，作为距离
        weights = np.exp(-distances) / np.sum(np.exp(-distances))  # 归一化权重
        weighted_diff = np.average(diffs, axis=0, weights=weights)
        weighted_differences[i] = weighted_diff
    # 返回的是按每个节点与其相邻节点特征向量的加权差并归一化权重后的特征矩阵
    return torch.tensor(weighted_differences).to(self.opt['device'])

  # 仅做差取做diffusion，固定系数
  def anisotropic_diffusion(self, x):
    # 获取节点数量
    n = max(max(self.edge_index[0]), max(self.edge_index[1])) + 1
    # 创建稀疏矩阵
    data = np.ones(len(self.edge_index[0]))
    row_indices, col_indices = self.edge_index.cpu()
    sparse_adj_matrix = coo_matrix((data, (row_indices, col_indices)), shape=(n, n))

    # 每个节点和其相邻节点特征向量做差，固定系数
    x = x.cpu().detach().numpy()
    weighted_differences = np.zeros_like(x)
    for i in range(sparse_adj_matrix.shape[0]):  # A*AT???
      neighbors = sparse_adj_matrix.getrow(i).indices  # 找出相邻节点的索引
      if len(neighbors) > 0:
        diffs = x[neighbors] - x[i]
        # distances = np.linalg.norm(diffs, axis=1)  # 计算差值的范数，作为距离
        # weights = np.exp(-distances) / np.sum(np.exp(-distances))  # 归一化权重
        # weighted_diff = np.average(diffs, axis=0, weights=weights)
        weighted_diff = np.sum(diffs, axis=0)
        weighted_differences[i] = weighted_diff
    # 返回的是按每个节点与其相邻节点特征向量的加权差并归一化权重后的特征矩阵
    return torch.tensor(weighted_differences).to(self.opt['device'])

  # 仅做差取做diffusion，1/(1+diff做二范数)加权
  def anisotropic_diffusion_g(self, x, kappa=1):
    # 获取节点数量
    n, d = x.shape
    # n = 2485

    # 创建稀疏矩阵
    data = np.ones(len(self.edge_index[0]))
    row_indices, col_indices = self.edge_index.cpu()
    sparse_adj_matrix = coo_matrix((data, (row_indices, col_indices)), shape=(n, n))

    # 每个节点和其相邻节点特征向量做差，加权系数
    x = x
    weighted_differences = torch.zeros_like(x)
    for i in range(sparse_adj_matrix.shape[0]):  # A*AT 矩阵化堆叠有问题，特征向量不是一维
      neighbors = sparse_adj_matrix.getrow(i).indices  # 找出相邻节点的索引
      if len(neighbors) > 0:
        diffs = x[neighbors] - x[i]
        distances = torch.linalg.norm(diffs, axis=1)  # 计算差值的范数，作为距离，  把distance设为0，评估一下weights是否是真的起作用
        weights = 1 / (1 + (distances / kappa) ** 2)
        weights = weights / torch.sum(weights)  # 加上归一化试试
        weights = weights.unsqueeze(1)
        weighted_diff = torch.sum(diffs * weights, axis=0)
        weighted_differences[i] = weighted_diff
    # 返回的是按每个节点与其相邻节点特征向量的加权差并归一化权重后的特征矩阵
    return weighted_differences

  # 仅做差取做diffusion，1/(1+diff做二范数)加权
  def anisotropic_diffusion_g_without_distance(self, x, kappa=1):
    # 获取节点数量
    n, d = x.shape  # n = 2485

    # 创建稀疏矩阵
    data = np.ones(len(self.edge_index[0]))
    row_indices, col_indices = self.edge_index.cpu()
    sparse_adj_matrix = coo_matrix((data, (row_indices, col_indices)), shape=(n, n))

    # 每个节点和其相邻节点特征向量做差，加权系数
    x = x
    weighted_differences = torch.zeros_like(x)
    for i in range(sparse_adj_matrix.shape[0]):  # A*AT 矩阵化堆叠有问题，特征向量不是一维
      neighbors = sparse_adj_matrix.getrow(i).indices  # 找出相邻节点的索引
      if len(neighbors) > 0:
        diffs = x[neighbors] - x[i]  # 4*80
        # distances = np.linalg.norm(diffs, axis=1)  # 计算差值的范数，作为距离，  把distance设为0，评估一下weights是否是真的起作用
        # weights = torch.ones(diffs.shape[0])
        # weights = weights / np.sum(weights)  # 加上归一化试试
        # weights = weights[:, np.newaxis]
        # weighted_diff = np.sum(diffs * weights, axis=0)
        weighted_diff = torch.mean(diffs, dim=0)  # 80
        weighted_differences[i] = weighted_diff
    # 返回的是按每个节点与其相邻节点特征向量的加权差并归一化权重后的特征矩阵
    return weighted_differences

  # 仅做差取做diffusion，1/(1+diff做二范数)加权
  def anisotropic_diffusion_g_matrix(self, x, kappa=1):
    n, d = x.shape
    x1 = x.unsqueeze(1).repeat(1, n, 1)  # torch.Size([2485, 2485, 80]) 行重复
    x2 = x.unsqueeze(0).repeat(n, 1, 1)  # torch.Size([2485, 2485, 80]) 列重复
    x_diff = x1 - x2  # torch.Size([2485, 2485, 80])
    # edge_index  torch.Size([2, 12623])

    # 矩阵化邻接矩阵 out of memory
    # A = torch.zeros((n, n), dtype=torch.int).to('cuda')  # torch.Size([2485, 2485])
    # A[edge_index[0, :], edge_index[1, :]] = 1
    # x_diff_A = A.unsqueeze(2) * x_diff  # out of memory
    # print(x_diff_A.shape)

    # 创建稀疏矩阵
    data = np.ones(len(self.edge_index[0]))
    row_indices, col_indices = self.edge_index.cpu()
    sparse_adj_matrix = coo_matrix((data, (row_indices, col_indices)), shape=(n, n))

    weighted_differences = torch.zeros_like(x)

    for i in range(n):
      neighbors = sparse_adj_matrix.getrow(i).indices
      if len(neighbors) > 0:
        x_i_diff = x_diff[i, neighbors, :]  # (4 * 80)
        distances = torch.linalg.norm(x_i_diff, axis=1)  # 计算差值的范数，作为距离，  把distance设为0，评估一下weights是否是真的起作用
        weights = 1 / (1 + (distances / kappa) ** 2)
        weights = weights / torch.sum(weights)  # 加上归一化试试
        weights = weights.unsqueeze(1)
        weighted_diff = torch.sum(x_i_diff * weights, axis=0)
        weighted_differences[i] = weighted_diff

    return weighted_differences  # torch.Size([2485, 80])


  def forward(self, t, x):  # the t param is needed by the ODE solver.
    if self.nfe > self.opt["max_nfe"]:
      raise MaxNFEException
    self.nfe += 1
    print(self.nfe)
    # torch.save(x, 'tensor.pt')
    # torch.save(self.edge_index, 'edge_index.pt')

    # # origin code
    # ax = self.sparse_multiply(x)  # shape=(2485, 80)
    # if not self.opt['no_alpha_sigmoid']:
    #   alpha = torch.sigmoid(self.alpha_train)
    # else:
    #   alpha = self.alpha_train
    #
    # f = alpha * (ax - x)
    # if self.opt['add_source']:
    #   f = f + self.beta_train * self.x0  #

    # #积分很慢
    ax = self.anisotropic_diffusion_g(x)
    # ax = self.anisotropic_diffusion_g_without_distance(x)
    if not self.opt['no_alpha_sigmoid']:
      alpha = torch.sigmoid(self.alpha_train)
    else:
      alpha = self.alpha_train
    f = alpha * ax

    # # 相邻节点直接做差，差值计算出来权重
    # alpha = torch.sigmoid(self.alpha_train)
    # weight_diff = self.weight_diff(x)
    # print(self.nfe)
    # f = 0.1 * weight_diff

    return f

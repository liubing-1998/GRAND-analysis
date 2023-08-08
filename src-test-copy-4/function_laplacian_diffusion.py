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
    self.d = nn.Parameter(torch.zeros(opt['hidden_dim']) + 1)  # 80��1
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

    # ��ȡ�ڵ�����
    n = max(max(self.edge_index[0]), max(self.edge_index[1])) + 1
    # ����ϡ�����
    data = np.ones(len(self.edge_index[0]))
    row_indices, col_indices = self.edge_index.cpu()
    sparse_adj_matrix = coo_matrix((data, (row_indices, col_indices)), shape=(n, n))

    # ����ÿ���ڵ��������ڽڵ����������ļ�Ȩ���һ��Ȩ��
    x = x.cpu().detach().numpy()
    weighted_differences = np.zeros_like(x)
    for i in range(sparse_adj_matrix.shape[0]):   # A*AT???
      neighbors = sparse_adj_matrix.getrow(i).indices  # �ҳ����ڽڵ������
      if len(neighbors) > 0:
        diffs = x[neighbors] - x[i]
        distances = np.linalg.norm(diffs, axis=1)  # �����ֵ�ķ�������Ϊ����
        weights = np.exp(-distances) / np.sum(np.exp(-distances))  # ��һ��Ȩ��
        weighted_diff = np.average(diffs, axis=0, weights=weights)
        weighted_differences[i] = weighted_diff
    # ���ص��ǰ�ÿ���ڵ��������ڽڵ����������ļ�Ȩ���һ��Ȩ�غ����������
    return torch.tensor(weighted_differences).to(self.opt['device'])


  def forward(self, t, x):  # the t param is needed by the ODE solver.
    if self.nfe > self.opt["max_nfe"]:
      raise MaxNFEException
    self.nfe += 1

    # # origin code
    # ax = self.sparse_multiply(x)
    # if not self.opt['no_alpha_sigmoid']:
    #   alpha = torch.sigmoid(self.alpha_train)
    # else:
    #   alpha = self.alpha_train
    #
    # f = alpha * (ax - x)
    # if self.opt['add_source']:
    #   f = f + self.beta_train * self.x0

    # ���ڽڵ�ֱ�������ֵ�������Ȩ��
    alpha = torch.sigmoid(self.alpha_train)
    weight_diff = self.weight_diff(x)
    print(self.nfe)
    f = 0.1 * weight_diff
    return f

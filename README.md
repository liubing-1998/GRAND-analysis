# GRAND-analysis

1、更改论文中ODEFunc的前向传播部分，更改代码在 .\src-test-copy-4/function_laplacian_diffusion.py 中40-61行。

forward函数中的69-84行

```python
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

        # 相邻节点直接做差，差值计算出来权重
        alpha = torch.sigmoid(self.alpha_train)
        weight_diff = self.weight_diff(x)
        print(self.nfe)
        f = 0.1 * weight_diff
        return f
```


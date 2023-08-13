# GRAND-analysis

#### 2023.08.13、

加上pytorch profile运行时间分析，在./src-test-copy-4/run_GNN.py中226行，调用在281行

```python
def analysis_run_time(model, feat, pos_encoding):
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            model(feat, pos_encoding)
            print(prof.key_averages().table(sort_by="cpu_time_total"))
            logging.info(prof.key_averages().table(sort_by="cpu_time_total"))

    with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            model(feat, pos_encoding)
            print(prof.key_averages().table(sort_by="cuda_time_total"))
            logging.info(prof.key_averages().table(sort_by="cuda_time_total"))
```

更改.\src-test-copy-4/function_laplacian_diffusion.py

```python
# 做差取做diffusion，1/(1+diff做二范数)加权，for循环实现在function_laplacian_diffusion.py中91行  
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


# 做差取做diffusion，distance设置为0的for循环实现在function_laplacian_diffusion.py中118行 
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

# 做差取做diffusion，1/(1+diff做二范数)加权，矩阵化实现在function_laplacian_diffusion.py中145行， out of memory 报错在155行
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
```









*****************************************************************************************************************************************************************************************************

#### 2023.08.08、

更改论文中ODEFunc的前向传播部分，更改代码在 .\src-test-copy-4/function_laplacian_diffusion.py 中40-61行。

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


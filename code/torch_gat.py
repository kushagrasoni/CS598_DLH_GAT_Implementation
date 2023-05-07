import numpy as np
import torch
import scipy.sparse

class GATHead(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias, in_drop=0.0, coef_drop=0.0, residual=False):
        super().__init__()
        self.bias1 = torch.nn.Parameter(bias)
        self.bias2 = torch.nn.Parameter(torch.zeros(out_dim))
        self.in_drop = in_drop
        self.coef_drop = coef_drop
        self.residual = residual

        self.conv1 = torch.nn.Conv1d(in_dim, out_dim, 1, bias=False)
        self.conv2 = torch.nn.Conv1d(out_dim, 1, 1)
        self.conv3 = torch.nn.Conv1d(in_dim, out_dim, 1)

    def forward(self, data):
        if self.in_drop != 0.0:
            data = torch.nn.functional.dropout(data, 1.0 - self.in_drop)

        feats = self.conv1(data.permute(0, 2, 1))

        f_1 = self.conv2(feats)
        logits = f_1 + f_1.permute(0, 2, 1)
        coefs = torch.nn.functional.softmax(torch.nn.functional.leaky_relu(logits) + self.bias1, dim=-1)

        if self.coef_drop != 0.0:
            coefs = torch.nn.functional.dropout(coefs, 1.0 - self.coef_drop)
        if self.in_drop != 0.0:
            feats = torch.nn.functional.dropout(feats, 1.0 - self.in_drop)

        vals = torch.matmul(coefs.float(), feats.permute(0, 2, 1))
        ret = vals + self.bias2

        if self.residual:
            if data.shape[-1] != ret.shape[-1]:
                ret = ret + self.conv3(data)
            else:
                ret = ret + data

        return ret

class GATLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, n_heads, bias, in_drop, coef_drop, residual, concat):
        super().__init__()
        self.heads = torch.nn.ParameterList([GATHead(in_dim, out_dim, bias, in_drop, coef_drop, residual) for i in range(n_heads)])
        self.concat = concat

    def forward(self, data):
        head_out = [self.heads[i](data) for i in range(len(self.heads))]
        if self.concat:
            return torch.concat(head_out, -1)
        return torch.sum(torch.stack(head_out, -1), dim=-1) / len(head_out)

class GAT(torch.nn.Module):
    def __init__(self, n_features, n_classes, bias):
        super().__init__()
        self.conv1 = GATLayer(n_features, 8, 8, bias, .6, .6, False, True)
        self.conv2 = GATLayer(64, n_classes, 1, bias, .6, .6, False, False)

    def forward(self, x):
        x = torch.nn.functional.elu(self.conv1(x))
        x = self.conv2(x)
        return torch.nn.functional.softmax(x, dim=-1)

# Taken directly from https://github.com/PetarV-/GAT/blob/master/utils/process.py
def adj_to_bias(adj, sizes, nhood=1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)

def planetoid_adj_to_petarv_adj(adj):
    size = adj.max() + 1
    petarv_adj = scipy.sparse.csr_array((size, size))
    for row in adj:
        petarv_adj[row[0], row[1]] = 1
    return petarv_adj

class GATLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, n_heads, dropout, concat=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.lin = torch.nn.Linear(in_dim, n_heads*out_dim)
        self.src_proj = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.Tensor(1, n_heads, out_dim)))
        self.trg_proj = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.Tensor(1, n_heads, out_dim)))
        self.dropout = torch.nn.Dropout(dropout)
        self.leakyRelu = torch.nn.LeakyReLU()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.concat = concat

    def forward(self, data, connectivity):
        data1 = self.lin(self.dropout(data)).view(-1, self.n_heads, self.out_dim)
        data2 = self.dropout(data1)
        #data = torch.nn.functional.dropout(self.lin(torch.nn.functional.dropout(data, self.dropout)), self.dropout).view(-1, self.n_heads, self.out_dim)
        src_scores = (data2 * self.src_proj).sum(dim=-1, keepdim=True).permute(1, 0, 2)
        trg_scores = (data2 * self.trg_proj).sum(dim=-1, keepdim=True).permute(1, 2, 0)
        scores = self.softmax(self.leakyRelu(src_scores + trg_scores) + connectivity)
        data3 = torch.bmm(scores.float(), data2.permute(1, 0, 2).float()).permute(1, 0, 2)
        # skip connection??

        if self.concat:
            data4 = data3.reshape(-1, self.n_heads*self.out_dim)
        else:
            data4 = data3.mean(dim=1)
        # bias?
        return data4

class GATCora(torch.nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.layer1 = GATLayer(n_features, 8, 8, .6, True)
        self.elu = torch.nn.ELU()
        self.layer2 = GATLayer(64, n_classes, 1, .6, False)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x, connectivity):
        x = self.elu(self.layer1(x, connectivity))
        return self.softmax(self.layer2(x, connectivity))

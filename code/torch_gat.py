import numpy as np
import torch

class GATHead(torch.nn.Module):
    def __init__(self, n_features, n_classes, bias, act, in_drop=0.0, coef_drop=0.0, residual=False):
        super().__init__()
        self.bias1 = bias
        self.bias2 = torch.zeros(n_features)
        self.act = act
        self.in_drop = in_drop
        self.coef_drop = coef_drop
        self.residual = residual

        self.conv1 = torch.nn.Conv1d(n_features, n_classes, 1, bias=False)
        self.conv2 = torch.nn.Conv1d(n_classes, 1, 1)
        self.conv3 = torch.nn.Conv1d(n_classes, 1, 1)
        self.conv4 = torch.nn.Conv1d(n_features, n_classes, 1)

    def forward(self, data):
        if self.in_drop != 0.0:
            data = torch.nn.functional.dropout(data, 1.0 - self.in_drop)

        feats = self.conv1(data)

        f_1 = self.conv2(feats)
        f_2 = self.conv3(feats)
        logits = f_1 + f2.swap_axes([0, 2, 1])
        coefs = torch.nn.functional.softmax(tf.nn.functional.leaky_relu(logits) + self.bias1, dim=-1)

        if self.coef_drop != 0.0:
            coefs = torch.nn.functional.dropout(coefs, 1.0 - self.coef_drop)
        if self.in_drop != 0.0:
            feats = torch.nn.functional.dropout(feats, 1.0 - self.in_drop)

        vals = torch.matmul(coefs, feats)
        ret = vals + self.bias2

        if self.residual:
            if data.shape[-1] != ret.shape[-1]:
                ret = ret + self.conv4(data)
            else:
                ret = ret + data

        return self.act(ret)

class GATLayer(torch.nn.Module):
    def __init__(self, n_features, n_classes, n_heads, bias, act, in_drop, coef_drop, residual, concat):
        super().__init__()
        self.heads = [GATHead(n_features, n_classes, bias, act, in_drop, coef_drop, residual) for i in range(n_heads)]
        self.concat = concat

    def forward(self, data):
        head_out = [self.heads[i](data) for i in range(len(self.heads))]
        if self.concat:
            return torch.concat(head_out, -1)
        return torch.sum(head_out) / len(head_out)

class GAT(torch.nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.conv1 = GATLayer(n_features, n_classes, 8, torch.zeros(())

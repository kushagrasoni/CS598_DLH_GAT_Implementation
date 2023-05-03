import torch_geometric
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from datetime import datetime
import numpy as np

dataset_name = 'cora'


class GATCora(torch.nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.conv1 = torch_geometric.nn.GATv2Conv(heads=8, out_channels=8, in_channels=in_channels)
        self.act1 = torch.nn.ELU()
        self.conv2 = torch_geometric.nn.GATv2Conv(heads=1, out_channels=n_classes, in_channels=64)
        self.act2 = torch.nn.Softmax(dim=1)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.act1(self.conv1(x, edge_index))
        x = self.act2(self.conv2(x, edge_index))
        return x


def execute_gat_model():
    start = datetime.now()
    dataset = Planetoid(root=f'../data/{dataset_name}', name=dataset_name)
    # Define model and optimizer
    model = GATCora(dataset.num_features, dataset.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    best_epoch = 0
    best_loss = 1e10
    patience = 100
    best_acc = 0.0
    # Train model
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        out = model(dataset.x, dataset.edge_index)
        # loss = F.nll_loss(out[dataset.train_mask], dataset.y[dataset.train_mask])
        loss = F.cross_entropy(out[dataset.train_mask], dataset.y[dataset.train_mask])
        loss.backward()
        optimizer.step()

        # Evaluate model
        model.eval()
        pred = model(dataset.x, dataset.edge_index).argmax(dim=1)
        correct = int(pred[dataset.train_mask].eq(dataset.y[dataset.train_mask]).sum().item())
        acc = correct / int(dataset.train_mask.sum())

        if (acc >= best_acc) or (loss <= best_loss):
            best_acc = np.max((acc, best_acc))
            best_epoch = np.max((epoch, best_epoch))
            best_loss = np.min((loss.detach().numpy(), best_loss))

        if epoch - best_epoch > patience:
            break

    # Test the model
    model.eval()
    out = model(dataset.x, dataset.edge_index)
    pred = out.argmax(dim=1)
    acc = pred[dataset.test_mask].eq(dataset.y[dataset.test_mask]).sum().item() / int(dataset.test_mask.sum())
    print('\n\n*****************************************************************************************************\n')
    print(f'                                         {dataset} ')
    print(f'                                         Total Epochs: 200')
    print(f'                                         Test Accuracy: {acc:.4f}')
    print(f'                                         Best Accuracy: {best_acc:.4f}')
    print(f'                                         Best Loss: {best_loss:.4f}')
    print(f'                                         Time Taken: {datetime.now() - start}')
    print('\n*****************************************************************************************************\n\n')

    return f'{acc: .4f}'


execute_gat_model()

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GAT
import sys

for dataset in ['citeseer', 'cora', 'pubmed']:
    # Load Citeseer dataset
    dataset = Planetoid(root=f'data/{dataset}', name=dataset)

    # Define model and optimizer
    model = GAT(
        in_channels=dataset.num_features,
        out_channels=dataset.num_classes,
        hidden_channels=8,
        num_layers=2,
        heads=8,
        dropout=0.6,
        act='elu',
        act_first=True
    )
    # {'PairNorm', 'GraphSizeNorm', 'HeteroLayerNorm', 'InstanceNorm', 'BatchNorm', 'DiffGroupNorm', 'GraphNorm', 'HeteroBatchNorm', 'MessageNorm', 'MeanSubtractionNorm', 'LayerNorm'}
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    # Train model
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        out = model(dataset.x, dataset.edge_index)
    #    print(out, out.argmax(1), dataset.y)
    #    input()
        #loss = F.nll_loss(out[dataset.train_mask], dataset.y[dataset.train_mask])
        loss = F.cross_entropy(out[dataset.train_mask], dataset.y[dataset.train_mask])
        loss.backward()
        optimizer.step()

        # Evaluate model
        model.eval()
        pred = model(dataset.x, dataset.edge_index).argmax(dim=1)
        correct = int(pred[dataset.train_mask].eq(dataset.y[dataset.train_mask]).sum().item())
        acc = correct / int(dataset.train_mask.sum())
        print(f'Epoch {epoch + 1:03d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}')


    # Test the model
    model.eval()
    out = model(dataset.x, dataset.edge_index)
    pred = out.argmax(dim=1)
    acc = pred[dataset.test_mask].eq(dataset.y[dataset.test_mask]).sum().item() / int(dataset.test_mask.sum())
    print(f'{dataset} Test Accuracy: {acc:.4f}')

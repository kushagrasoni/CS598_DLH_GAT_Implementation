
# CS598 DLH - Project - Reproduce & Experiment Graph Attention Networks

## Checklist

1. [ ] architecture differences (different activation functions on different layers, etc.)
   1. [ ] GATConv, highly optimized linear algebra, better performance
   2. [ ] Pure Python, lot more code, more work
2. [ ] dropout in the paper seems to be different from regular dropout
3. [ ] early stopping
4. [ ] PPI training seemed to have some specificities to it (something about using two graphs)
5. [ ] L2 regularization
6. [ ] ablation studies
   1. [ ] Replacing the self-attention mechanism with other metrics, such as constants, random weights, and Pearson and Spearman correlation coefficients.
   2. [ ] Replacing the shared weight matrix W with the encoding phase of an autoencoder, by first training an autoencoder on the data. 
   3. [ ] Getting rid of the shared weight matrix W
   4. [ ] Getting rid of the vector a^T
7. [ ] avoid overfitting (might be addressed by early stopping)
8. [ ] GPU enablement


##Things We tried but didn't work

We tried to use the softmax function in below ways but that made the accuracy worst.

```
def forward(self, x, edge_index):
x = self.act1(self.act2(self.conv1(x, edge_index)))
x = self.act2(self.conv2(x, edge_index))
return x
```


```
def forward(self, x, edge_index):
x = self.act1(F.softmax(self.conv1(x, edge_index), dim=1))
x = self.act2(self.conv2(x, edge_index))
return x
```
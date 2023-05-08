# Spring 2023 CS598 DL4H: Graph Attention Networks Reproducibility Project

This repository attempts to reproduce the Graph Attention Networks (GAT) paper using 
Pytorch, and includes experiments using variations of the model architectures used in the original GAT paper.

## Project Setup
To get the project code run below command:
```setup
git clone https://github.com/kushagrasoni/CS598_DLH_GAT_Implementation.git
```

### Requirements & Dependencies
To install required Python libraries, 
* Navigate to the project root directory
* Execute below commands to create a python virtual environment, and install the required dependencies from the 
  requirements.txt file
```setup
python -m venv <venv_name>
python -m pip install -r requirements.txt
```

## Project Overview & Implementation

### Jupyter Notebook

Once you've set up your environment, follow the below steps to view all the various GAT implementation by our team
using the Jupyter Notebook.
* Navigate to the `/code` directory of the cloned project
* Open the `jupyter notebook` named `GAT_Implementation_Notebook.ipynb`.

NOTE: We have also kept all the draft and workspace code implementation for 3 datasets in separate Notebooks
under `/code/workspace`

### Model Implementation and Description
* We tried a total of 5 GAT implementations, four of which were our own, and one of which is from the Pytorch Geometric library.
* One of our implementations was strongly influenced by the paper's authors' implementation.
* All datasets from the paper are publicly available in multiple locations,
including the Pytorch Geometric library. We have been running our
experiments locally. 
* Despite the datasets being relatively small, the PPI dataset turned out to be sufficiently large to take a prohibitively long
time to run. We therefore ended up not having any results for the PPI dataset.

### Results & Benchmarkings

We did a Benchmarking test on our model by executing each dataset 50 times, where at every iteraton
the training is conducted for 200 epochs.
We received the follow performance metrics:

| Dataset   | GAT Type      | Mean     | Std.    |
|-----------|---------------|----------|---------|
| Cora      | GAT           | 78.99%   | 1.13%   |
| Cora      | GATConv       | 78.86%   | 1.15%   |
| Cora      | GATv2Conv     | 78.53%   | 1.04%   |
| Citeseer  | GAT           | 67.01%   | 1.43%   |
| Citeseer  | GATConv       | 66.83%   | 1.41%   |
| Citeseer  | GATv2Conv     | 66.65%   | 1.47%   |
| Pubmed    | GAT           | 77.73%   | 1.11%   |
| Pubmed    | GATConv       | 77.38%   | 1.32%   |
| Pubmed    | GATv2Conv     | 77.73%   | 1.22%   |

## Contribution

* This reproducibility project was created as part of the CS598 Deep Learning for Healthcare class at UIUC in the
  Spring 2023 semester.
semeseter. Contributions are not welcome.

`Team Name: GATors`

`Team Members: 
Soni, Kushagra
Valle-Mena, Ricardo Ruy`


import os
from timer import timer

wd = os.getcwd()

os.chdir(wd)

num_tests = 50  #, 20, 30, 40, 50]

datasets = ['citeseer', 'pubmed', 'cora']  #'cora', 'citeseer', 'pubmed']

list_gat_type = ['GATv2Conv', 'GATConv', 'GAT']

# int_runs = [1, 2, 4]

if __name__ == '__main__':
    uid = '01'
    for dataset in datasets:
        print(f"Running for Dataset {dataset}, {num_tests} times")

        timer(num_tests, uid, dataset, list_gat_type)

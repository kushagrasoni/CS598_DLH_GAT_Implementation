from pathlib import Path
import os
from timer import timer

wd = os.getcwd()

os.chdir(wd)

int_runs = [5, 10] #, 20, 30, 40, 50]

datasets = ['cora'] #, 'citeseer', 'pubmed']

gat_type = 'GATv2Conv'

# int_runs = [1, 2, 4]

if __name__ == '__main__':
    uid = 'gat_01'
    for dataset in datasets:
        for n in int_runs:
            num_tests = str(n)
            print(f"Running for {n}")

            timer(num_tests, uid, dataset, gat_type)

    os.chdir(wd)

    import json
    import pandas as pd

    recs = []

    for p in Path('.').rglob(f'run_result_*_{uid}.json'):
        with open(p) as f:
            d = json.load(f)
            recs.append(d)

    df = pd.DataFrame.from_records(recs)

    df['time'] = df['mean'] * df['num_tests']

    p_df = df.pivot_table(columns=['dataset', 'mean', 'num_tests', 'gat_type'])
    p_df.to_csv(f'benchmarking_pivot_{uid}.csv', index=True)
    df.to_csv(f'benchmarking_{uid}.csv', index=False)

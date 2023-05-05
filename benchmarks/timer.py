from random import randint
import numpy as np
import json
import pandas as pd


from GATConv_Cora import execute_gat_model


def timer(num_tests, uid, dataset, list_gat_type):
    num_tests = int(num_tests)
    all_accuracy = []
    accuracy_records = []

    for gat_type in list_gat_type:
        for i in range(num_tests):
            seed = randint(0, 100000)
            print(f"GAT Type {gat_type} ; Iteration {i+1} ; Seed {seed}")
            # start_time = timeit.default_timer()
            # This is the where we call the target function
            accuracy = execute_gat_model(gat_type, dataset, seed)
            # end_time = timeit.default_timer()
            # t = (end_time - start_time)

            all_accuracy.append(accuracy)

            result = {
                "dataset": dataset,
                "gat_type": gat_type,
                "seed": seed,
                "accuracy": float(accuracy)
            }

            accuracy_records.append(result)

        with open(f'Average_Test_Accuracy_Result_{dataset}_{num_tests}_{uid}.json', 'a') as file:
            data = np.array(all_accuracy)
            data = data.astype(float)
            result_all = {
                "mean": np.mean(data),
                "min": np.min(data),
                "max": np.max(data),
                "std": np.std(data),
                "num_tests": num_tests,
                "dataset": dataset,
                "gat_type": gat_type,
            }
            result_all_json = json.dumps(result_all)

            file.write(result_all_json)

    df = pd.DataFrame.from_records(accuracy_records)

    df.to_csv(f'Test_Accuracy_Results_{dataset}_{num_tests}_{uid}.csv', mode='a', index=False)


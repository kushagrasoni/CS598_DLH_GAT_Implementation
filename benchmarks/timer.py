from random import randint
import numpy as np
import json
import pandas as pd


from GAT_Cora import execute_model_cora
from GAT_Citeseer import execute_model_citeseer
from GAT_Pubmed import execute_model_pubmed


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
            if dataset == 'cora':
                accuracy, time_taken, best_accuracy, best_loss = execute_model_cora(gat_type, seed)
            elif dataset == 'citeseer':
                accuracy, time_taken, best_accuracy, best_loss = execute_model_citeseer(gat_type, seed)
            else:
                accuracy, time_taken, best_accuracy, best_loss = execute_model_pubmed(gat_type, seed)
            # end_time = timeit.default_timer()
            # t = (end_time - start_time)

            all_accuracy.append(accuracy)

            result = {
                "dataset": dataset,
                "gat_type": gat_type,
                "seed": seed,
                "time_taken": time_taken,
                "best_accuracy": best_accuracy,
                "best_loss": best_loss,
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


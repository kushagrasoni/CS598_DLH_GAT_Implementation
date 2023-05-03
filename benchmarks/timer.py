import numpy as np
import json
from GATConv_Cora import execute_gat_model


def timer(num_tests, uid, dataset, gat_type):
    accuracy = []
    num_tests = int(num_tests)
    for i in range(num_tests):
        print(f"Iter {i+1}")
        # start_time = timeit.default_timer()
        # This is the where we call the target function
        acc = execute_gat_model()
        # end_time = timeit.default_timer()
        # t = (end_time - start_time)
        accuracy.append(acc)

    data = np.array(accuracy)
    data = data.astype(float)
    result = {
        "mean": np.mean(data),
        "min": np.min(data),
        "q25": round(np.quantile(data, .25), 4),
        "median": round(np.quantile(data, .50), 4),
        "q75": round(np.quantile(data, .75), 4),
        "max": np.max(data),
        "std": np.std(data),
        "num_tests": num_tests,
        "dataset": dataset,
        "gat_type": gat_type,
    }

    result_json = json.dumps(result)
    with open(f'run_result_{dataset}_{num_tests}_{gat_type}_{uid}.json', 'w') as file:
        file.write(result_json)

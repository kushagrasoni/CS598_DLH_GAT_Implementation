import numpy as np
import json
from GATConv_Cora import execute_gat_model


def timer(num_tests, uid, dataset, gat_type):
    accuracy = []
    num_tests = int(num_tests)
    for i in range(num_tests):
        print(f"Iter {i}")
        # start_time = timeit.default_timer()
        # This is the where we call the target function
        acc = execute_gat_model()
        # end_time = timeit.default_timer()
        # t = (end_time - start_time)
        accuracy.append(acc)

    d = np.array(accuracy)
    result = {
        "mean": np.mean(d),
        "min": np.min(d),
        "q25": np.quantile(d, .25),
        "median": np.quantile(d, .50),
        "q75": np.quantile(d, .75),
        "max": np.max(d),
        "std": np.std(d),
        "num_tests": num_tests,
        "dataset": dataset,
        "gat_type": gat_type,
    }

    result_json = json.dumps(result)
    with open(f'run_result_{dataset}_{num_tests}_{gat_type}_{uid}.json', 'w') as file:
        file.write(result_json)

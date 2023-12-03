"""Usage: After parameter search, 
    - use generate_rerun_script.py to generate the retrain script: rerun_{experiment_name}.sh
    - sh rerun_{experiment_name}.sh
    - python3 dump_results.py --tune_dir $YOUR_TUNE_DIR (runs/{experiment_name}...)
"""
import argparse
import collections
import glob
import json
import os

import numpy as np
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--tune_dir", help="Path to ray tune directory.")
    parser.add_argument("--output_path", help="Path to the tune results.")
    args, _ = parser.parse_known_args()

    # tune_dir, tune_dir/
    exp_name = args.tune_dir.split("/")[-1] or args.tune_dir.split("/")[-2]
    exp_stat_path = glob.glob(f"runs/{exp_name}/experiment_state*.json")[0]
    print(exp_stat_path)
    exp_state = json.load(open(exp_stat_path, "r"))
    params = json.loads(exp_state["checkpoints"][0])["evaluated_params"].keys()

    error_cnt = 0
    data = collections.defaultdict(list)

    for file in glob.glob(f"runs/{exp_name}/**/logs.json", recursive=True):
        try:
            with open(file, "r") as f:
                log = json.load(f)
                config = log["config"]
                idx = np.argmax([values[config["val_metric"]] for values in log["val"]])

                # recursively get params in network config (e.g., network_config/embed_dropout)
                for param in params:
                    value = config
                    for key in param.split("/"):
                        value = value.get(key, None)
                    data[param].append(value)
                for metric in config["monitor_metrics"]:
                    data[metric].append(log["val"][idx][metric])
        except:
            # skip error logs with only "val", use the complete result in the retrain subfolder
            error_cnt += 1

    output_path = args.output_path or f"{exp_name}_results.csv"
    df = pd.DataFrame(data)
    df.sort_values(by="Micro-F1", ascending=False).to_csv(output_path)
    print(f"Write {len(df)} results to {output_path}.")

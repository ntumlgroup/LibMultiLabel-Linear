import glob
import sys, json
import argparse
import pandas as pd
from pathlib import Path

def make_csv_table(array):
    df = pd.DataFrame(array[1:], columns=array[0])
    df_mean = df.groupby('model').mean().mul(100).applymap('{:2.2f}'.format).reset_index()

    df_std = df.groupby('model').std().mul(100).applymap('{:.2f}'.format).reset_index()
    df = df_mean.merge(df_std, on="model", suffixes=["_avg", "_std"])
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root', default="runs")
    parser.add_argument('-m', '--metrics',nargs="*", default=["Macro-F1", "Micro-F1", "P@5", "nDCG@5"])
    parser.add_argument('-t', '--tasks',nargs="*", default=["EUR-Lex", "MIMIC-50", "ECtHRA"])
    parser.add_argument('-s', '--search', action='store_true')
    parser.add_argument('-c', '--csv', action='store_true')
    parser.add_argument('-p', '--paper', action='store_true')
    args = parser.parse_args()

    json_files = sorted(glob.glob(f"{args.root}/*/trial_best_params/logs.json")) if args.search else sorted(glob.glob(f"{args.root}/*/logs.json"))
    tasks = dict()
    undone = []
    for i in args.tasks:
        tasks[i] = [["model"]]
        tasks[i][0] += [i for i in args.metrics]
    for file in json_files:
        with open(file, 'r') as r:
            log = json.load(r)
        # check if the experiment is completed
        if not log.get("test", False) or not log.get("config", False):
            undone.append(file)
            continue
        # check if the task is required.
        task = log["config"]["data_name"]
        if task not in args.tasks:
            continue
        # get model name (config name)
        model_name = Path(log["config"]["config"]).stem
        # get required metrics
        score = log["test"][0]
        row = [model_name]
        for key, value in score.items():
            row.append(round(value, 4)) if key in args.metrics else None
        tasks[task].append(row)
    print(undone)
    if not args.paper:
        for task in tasks.keys():
            print(f"{task}:")
            print(make_csv_table(tasks[task]).to_csv(index=False, header=True))
    else:
        results = []
        replace_dict = {
            "bilstm_": "",
            "cnn_": "",
            "tanhW_": "\eqref{eq:tanhW}\_",
            "tanh_":"\eqref{eq:cnn_lwan}\_",
            "vanilla_":"\eqref{eq:vanilla-dot-product}\_",
            "mlp_tune":"\eqref{eq:mlp}",
            "_tune":"\eqref{eq:output_vanilla}"
        }
        tasks_list = list(tasks.keys())
        for task in tasks_list:
            results.append(make_csv_table(tasks[task]))
        task_p = r" & \bfseries ".join(tasks_list)
        table = ""
        table += r"& \bfseries " + task_p + r"\\" + "\n"
        table += r"\bfseries Model & & Micro-F1 &  \\ \midrule BiLSTM \\"
        for i in range(len(results[0][0:6])):
            row = []
            model_name = str(results[0].loc[i, "model"])
            for key, val in replace_dict.items():
                model_name = model_name.replace(key, val)
            row.append(model_name)
            for result in results:
                mean = result.loc[i, f"{args.metrics[0]}_avg"]
                std = result.loc[i, f"{args.metrics[0]}_std"]
                text = f"{mean} $\pm$ {std}"
                if mean == result[f"{args.metrics[0]}_avg"][0:6].max():
                    row.append("\B{" + text + "}")
                elif mean == result[f"{args.metrics[0]}_avg"][0:6].min():
                    
                    row.append(r"\underline{" + text + "}")
                else:
                    row.append(text)
            table += " & ".join(row) + r"\\" + "\n"
        table += r"\midrule" + "\n"
        row = [r"\bfseries Max-min"]
        for result in results:
            max = result[f"{args.metrics[0]}_avg"][0:6].max()
            min = result[f"{args.metrics[0]}_avg"][0:6].min()
            val = float(max)-float(min)
            val = round(val, 2)
            row.append(str(val))
        table += " & ".join(row) + "\n"
        table += r"\midrule" + "\n"
        table += r"CNN \\" + "\n"
        for i in range(len(results[0][6:12])):
            row = []
            model_name = str(results[0].loc[i, "model"])
            for key, val in replace_dict.items():
                model_name = model_name.replace(key, val)
            row.append(model_name)
            for result in results:
                mean = result.loc[i, f"{args.metrics[0]}_avg"]
                std = result.loc[i, f"{args.metrics[0]}_std"]
                text = f"{mean} $\pm$ {std}"
                if mean == result[f"{args.metrics[0]}_avg"][6:12].max():
                    row.append("\B{" + text + "}")
                elif mean == result[f"{args.metrics[0]}_avg"][6:12].min():
                    
                    row.append(r"\underline{" + text + "}")
                else:
                    row.append(text)
            table += " & ".join(row) + r"\\" + "\n"
        table += r"\midrule" + "\n"
        row = [r"\bfseries Max-min"]
        for result in results:
            max = result[f"{args.metrics[0]}_avg"][6:12].max()
            min = result[f"{args.metrics[0]}_avg"][6:12].min()
            val = float(max)-float(min)
            val = round(val, 2)
            row.append(str(val))
        table += " & ".join(row) + "\n"
        table += r"\bottomrule" + "\n"     

        print(table)

            
if __name__ == "__main__":
    main()
import glob
import sys, json
import argparse
import pandas as pd

def make_markdown_table(array):

    """ Input: Python list with rows of table as lists
               First element as header. 
        Output: String to put into a .md file 
        
    Ex Input: 
        [["Name", "Age", "Height"],
         ["Jake", 20, 5'10],
         ["Mary", 21, 5'7]] 
    """


    markdown = "\n" + str("| ")

    for e in array[0]:
        to_add = " " + str(e) + str(" |")
        markdown += to_add
    markdown += "\n"

    markdown += '|'
    for i in range(len(array[0])):
        markdown += str("-------------- | ")
    markdown += "\n"

    for entry in array[1:]:
        markdown += str("| ")
        for e in entry:
            to_add = str(e) + str(" | ")
            markdown += to_add
        markdown += "\n"

    return markdown + "\n"

def make_csv_table(array):
    df = pd.DataFrame(array[1:], columns=array[0])
    df_mean = df.groupby('model').mean().mul(100).round(2).reset_index()
    df_std = df.groupby('model').std().mul(100).round(2).reset_index()
    df = df_mean.merge(df_std, on="model", suffixes=["_avg", "_std"])
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root', default="runs")
    parser.add_argument('-m', '--metrics',nargs="*", default=["Macro-F1", "Micro-F1", "P@5", "nDCG@5"])
    parser.add_argument('-t', '--tasks',nargs="*", default=["EUR-Lex", "MIMIC-50", "ECtHRA"])
    parser.add_argument('-c', '--csv', action='store_true')
    parser.add_argument('-s', '--search', action='store_true')
    parser.add_argument('-p', '--paper', action='store_true')
    parser.add_argument('-d', '--des', type=str, default="trial_best_params")

    args = parser.parse_args()
    json_files = sorted(glob.glob(f"{args.root}/*/{args.des}/logs.json")) if args.search else sorted(glob.glob(f"{args.root}/*/logs.json"))
    tasks = dict()
    for i in args.tasks:
        tasks[i] = [["model"]]
        tasks[i][0] += [i for i in args.metrics]
    for file in json_files:
        with open(file, 'r') as r:
            log = json.load(r)
        if log["config"].get("custom_run_name", False):
            # print(file)
            task = "".join(log["config"]["result_dir"].split("_")[0])
            task = task.replace("runs/", "")
            model_name = "_".join(log["config"]["result_dir"].split("_")[1:-1])
        else:
            # breakpoint()
            task = log["config"]["data_name"]
            if args.des == "results":
                model_name = "_".join(log["config"]["result_dir"].split("_")[1:-1])
            else:
                model_name = "_".join(log["config"]["run_name"].split("_")[1:-1])
        if "test" not in log.keys():
            print(file)
            continue
        score = log["test"][0]
        row = []
        row.append(model_name)
        for i in score.keys():
            if i in args.metrics:
                row.append(round(score[i], 4))
        # print(task)
        tasks[task].append(row)
    
    if not args.paper:
        for task in tasks.keys():
            print(f"{task}:")
            if args.csv:
                print(make_csv_table(tasks[task]).to_csv(index=False, header=True))
            else:
                print(make_markdown_table(tasks[task]))    
    else:
        results = []
        for task in tasks.keys():
            print(task)
            results.append(make_csv_table(tasks[task]))
        for rows in zip(*[i.iterrows() for i in results]):
            row_pair = []
            for i in rows:
                row_pair.append((" \pm ").join([str(i[1][j]) for j in range(1, len(i[1]))]))
            print("& " + (" & ").join(row_pair))
if __name__ == "__main__":
    main()
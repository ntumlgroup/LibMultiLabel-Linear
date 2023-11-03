import glob
import sys, json
import argparse

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root', default="runs")
    parser.add_argument('-m', '--metrics', default=["Macro-F1", "Micro-F1", "P@5", "nDCG@5"])
    parser.add_argument('-t', '--tasks', default=["EUR-Lex", "MIMIC-50"])

    args = parser.parse_args()

    json_files = sorted(glob.glob(f"{args.root}/*/trial_best_params/logs.json"))
    tasks = dict()
    for i in args.tasks:
        tasks[i] = [["model"]]
        tasks[i][0] += [i for i in args.metrics]
    
    for file in json_files:
        with open(file, 'r') as r:
            log = json.load(r)
        task = log["config"]["data_name"]
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
        tasks[task].append(row)
    
    for task in tasks.keys():
        print(f"{task}:")
        md = make_markdown_table(tasks[task])
        print(md)    

if __name__ == "__main__":
    main()

    

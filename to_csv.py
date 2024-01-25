import json, csv, glob

files = sorted(glob.glob("/home/gordon1109/init_test/runs/*"))

with open("init_result.csv", "w") as writer:
    csv_writer = csv.writer(writer)
    for file in files:
        log = json.load(open(f"{file}/logs.json"))
        metric = log["config"]["val_metric"]
        csv_writer.writerow([log["config"]["data_name"], log["config"]["model_name"], metric, log["test"][0][metric]])
import pandas

with open("all_log") as F:
    all_log = []
    for f in F:
        row = f.split("\n")[0]
        if "==" in row:
            tmp = dict()
            name = row.split("/")[-1]
            name = name.split(".log")[0]
            tmp["name"] = name
            hyper_param = name.split("_")
            tmp["num_model"] = int(hyper_param[0].split("-")[-1])
            tmp["dataset"] = hyper_param[1]
            tmp["seed"] = int(hyper_param[2].split("=")[1])
            tmp["K"] = int(hyper_param[3].split("=")[1])
            tmp["beam_width"] = int(hyper_param[4].split("=")[1])
            tmp["sample_rate"] = float(hyper_param[5].split("=")[1])
        elif "labels:" in row:
            if "all" in row:
                row = row.split("labels:")[1]
                row = row.split("{")[-1]
                row = row.split("}")[0]
                metrics = row.split(",")
                tmp["all-P@1"] = float(metrics[0].split(": ")[-1])
                tmp["all-P@3"] = float(metrics[1].split(": ")[-1])
                tmp["all-P@5"] = float(metrics[2].split(": ")[-1])
            elif "sub" in row:
                row = row.split("labels:")[1]
                row = row.split("{")[-1]
                row = row.split("}")[0]
                metrics = row.split(",")
                tmp["sub-P@1"] = float(metrics[0].split(": ")[-1])
                tmp["sub-P@3"] = float(metrics[1].split(": ")[-1])
                tmp["sub-P@5"] = float(metrics[2].split(": ")[-1])
                all_log += [tmp]

all_log = pandas.DataFrame.from_dict(all_log)
all_log = all_log[all_log["K"]==100]
#all_log = all_log[all_log["K"]==20]
all_log = all_log[all_log["sample_rate"]>1.0]
all_log = all_log[all_log["beam_width"]==10]
#all_log = all_log[all_log["sample_rate"]==0.3]
#all_log = all_log[all_log["sample_rate"]>0.2]
all_log = all_log.drop("name", axis=1)
#all_log = all_log.drop("dataset", axis=1)
print(all_log)

mean = all_log.groupby(["num_model", "K", "dataset", "beam_width", "sample_rate"]).mean()*100
mean = mean.drop("seed", axis=1)
mean = mean.round(2)
#print(mean)
        
#std = all_log.groupby(["num_model", "K", "dataset", "beam_width", "sample_rate"]).std()*100
#std = std.drop("seed", axis=1)
#std = std.round(2)
#print(std)

#mean_latex = mean.to_latex()
#print(mean_latex)
#std_latex = std.to_latex()
#print(std_latex)

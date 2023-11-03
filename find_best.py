import glob, os, json

files = glob.glob("/work/d11922012/gordon/mlgroup/runs/EUR-Lex_cnn_tanhW_tune_20231101195659/*/logs.json")
maxi_file = None
global_maxi = 0
for file in files:
    with open(file, "r") as r:
        log = json.load(r)
    maxi = 0
    for idx, i in enumerate(log["val"]):
        maxi = max(i["P@5"], maxi)
    print(maxi)
    if maxi > global_maxi:
        global_maxi = maxi
        maxi_file = file
print(maxi_file)
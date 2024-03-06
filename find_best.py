import glob, os, json, argparse



import json, glob, argparse, os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--metric', default="Micro-F1")
    parser.add_argument('-r', '--root')
    args = parser.parse_args()

    files = glob.glob(f"{args.root}/*/logs.json")
    maxi_file = None
    global_maxi = 0
    for file in files:
        with open(file, "r") as r:
            log = json.load(r)
        maxi = 0
        if not log.get("val", False):
            # print(file)
            continue
        for idx, i in enumerate(log["val"]):
            maxi = max(i[f"{args.metric}"], maxi)
        if maxi > global_maxi:
            global_maxi = maxi
            maxi_file = file
    print(maxi_file)

if __name__ == "__main__":
    main()
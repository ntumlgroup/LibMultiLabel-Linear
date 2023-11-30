import argparse
import glob
import json
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--tune_dir", help="Path to ray tune directory.")
    args, _ = parser.parse_known_args()

    # tune_dir, tune_dir/
    exp_name = args.tune_dir.split("/")[-1] or args.tune_dir.split("/")[-2]
    trial_dirs = [os.path.dirname(f) for f in glob.glob(
        f"{args.tune_dir}/*/error.txt")]

    scripts = ""
    for trial_dir in trial_dirs:
        params_file = os.path.join(trial_dir, "params.json")
        config = json.load(open(params_file, "r"))
        config["result_dir"] = trial_dir

        # dump rerun config to trial dir
        config_path = f"{trial_dir}/params_rerun.json"
        json.dump(config, open(config_path, "w"))
        print(f"Write new config to {config_path}")
        scripts += f"python3 main.py --config {config_path}\n"

    print(f"Number of error trials: {len(trial_dirs)}")
    with open(f"rerun_{exp_name}.sh", "w") as f:
        f.write(scripts)

    print(f"Write script to rerun_{exp_name}.sh.")

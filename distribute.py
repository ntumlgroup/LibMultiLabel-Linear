import argparse
import os
import pathlib
import shlex
import shutil
import subprocess

import numpy as np
from tqdm import tqdm

import sshutils
from libmultilabel import linear

parser = argparse.ArgumentParser(
    add_help=False,
    description="distribute 1vsrest with ssh",
)

parser.add_argument("--hosts", type=str, nargs="+", help="Hosts to distribute on")
parser.add_argument(
    "--subdivision", type=int, default=1, help="Number of sub-divisions of work per host (default: %(default)s)"
)
parser.add_argument("--tmp_dir", type=str, default="tmp/result", help="Temporary directory name (default: %(default)s)")
parser.add_argument("--model_dir", type=str, default="./model", help="Path to resulting model (default: %(default)s)")
parser.add_argument("--result_dir", type=str, help="Do NOT use, result_dir cannot be specified")
parser.add_argument("--no_tqdm", action="store_true", help="Run without tqdm")
parser.add_argument("--mmap", action="store_true", help="Don't keep weights in memory, but memmap it instead")
parser.add_argument("-h", "--help", action="help", help="See this")

args, passthrough_args = parser.parse_known_args()
if args.result_dir is not None:
    raise ValueError("do NOT use result_dir")
hosts = args.hosts
tmp_dir = args.tmp_dir
subdivision = args.subdivision
model_dir = args.model_dir
disable_tqdm = args.no_tqdm
mmap = args.mmap

div = len(hosts) * subdivision
cmd = f'python main.py {" ".join(map(lambda x: shlex.quote(x), passthrough_args))}'
cmds = [f"{cmd} --label_subrange {i/div} {(i+1)/div} --result_dir {tmp_dir}" for i in range(div)]

print(f"Running {div} jobs on {len(hosts)} hosts.", flush=True)
handlers = sshutils.propogate_signal()
sshutils.distribute(cmds, hosts)
sshutils.propogate_signal(handlers)

cwd = sshutils.home_relative_cwd()
print("Copying from hosts")
pathlib.Path(f"{tmp_dir}").mkdir(parents=True, exist_ok=True)
for host in tqdm(hosts, disable=disable_tqdm):
    subprocess.call(f'scp -qr "{host}:{cwd}/{tmp_dir}" "{tmp_dir}/{host}"', shell=True, executable="/bin/bash")
    sshutils.execute(f"rm -r {tmp_dir}", [host])

print("Reconstructing model")
pbar = tqdm(total=div, disable=disable_tqdm)
weights = None
bias = None
for root, _, files in os.walk(tmp_dir):
    for file in files:
        if file != "linear_pipeline.pickle":
            continue
        preprocessor, model = linear.load_pipeline(f"{root}/{file}")
        if weights is None:
            num_labels = len(preprocessor.binarizer.classes_)
            num_features = model.weights.shape[0]
            if mmap:
                pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)
                weights = np.memmap(f"{model_dir}/weights.dat", shape=(num_features, num_labels), dtype="d", mode="w+")
            else:
                weights = np.zeros((num_features, num_labels))
        if bias is None:
            bias = model.bias

        weights[:, model.subset] = model.weights
        pbar.update()
pbar.close()


combined_model = {
    "-B": bias,
    "threshold": 0,
}

if mmap:
    combined_model.mmap = {"shape": (num_features, num_labels), "dtype": "d"}
else:
    combined_model.weights = np.asmatrix(weights)

linear.save_pipeline(model_dir, preprocessor, combined_model)
shutil.rmtree(tmp_dir)

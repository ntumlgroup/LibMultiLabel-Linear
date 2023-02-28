import argparse
import os
import pathlib
import shlex
import shutil
import subprocess
import time

import numpy as np
from tqdm import tqdm

from libmultilabel import linear
from rungrid import Grid

parser = argparse.ArgumentParser(
    add_help=False,
    description='distribute 1vsrest with ssh',
)

parser.add_argument('--hosts', type=str, nargs='+',
                    help='Hosts to distribute on')
parser.add_argument('--subdivision', type=int, default=1,
                    help='Number of sub-divisions of work per host (default: %(default)s)')
parser.add_argument('--tmp_dir', type=str, default='tmp/result',
                    help='Temporary directory name (default: %(default)s)')
parser.add_argument('--model_dir', type=str, default='./model',
                    help='Path to resulting model (default: %(default)s)')
parser.add_argument('--result_dir', type=str,
                    help='Do NOT use, result_dir cannot be specified')
parser.add_argument('--no-tqdm', action='store_true',
                    help='Run without tqdm')
parser.add_argument('-h', '--help', action='help',
                    help="See this")

args, passthrough_args = parser.parse_known_args()
if args.result_dir is not None:
    raise ValueError('do NOT use result_dir')
hosts = args.hosts
tmp_dir = args.tmp_dir
subdivision = args.subdivision
model_dir = args.model_dir
disable_tqdm = args.no_tqdm

div = len(hosts) * subdivision
cmd = f'python main.py {" ".join(map(lambda x: shlex.quote(x), passthrough_args))}'
jobs = [
    f'{cmd} --label_subrange {i/div} {(i+1)/div} --result_dir {tmp_dir}'
    for i in range(div)
]

grid = Grid(hosts, jobs)
grid.go()

start = time.time()

cwd = os.getcwd()
home = os.path.expanduser('~')
if cwd.startswith(home):
    cwd = cwd[len(home):]
    if cwd.startswith(os.sep):
        cwd = cwd[1:]

print('Copying from hosts')
pathlib.Path(f'{tmp_dir}').mkdir(parents=True, exist_ok=True)
for host in tqdm(hosts, disable=disable_tqdm):
    subprocess.call(f'scp -qr "{host}:{cwd}/{tmp_dir}" "{tmp_dir}/{host}"',
                    shell=True, executable='/bin/bash')
    subprocess.call('ssh {} "rm -r {}"'.format(host, shlex.quote(f'{cwd}/{tmp_dir}')),
                    shell=True, executable='/bin/bash')

print('Loading models')
pbar = tqdm(total=div, disable=disable_tqdm)
labels = set()
models = []
for root, _, files in os.walk(tmp_dir):
    for file in files:
        if file != 'linear_pipeline.pickle':
            continue
        preprocessor, model = linear.load_pipeline(f'{root}/{file}')
        labels.update(model['subset'].tolist())
        models.append(model)
        pbar.update()
pbar.close()

print('Reconstructing model')
bias = models[0]['-B']
weights = np.zeros((models[0]['weights'].shape[0], len(labels)), order='F')
for model in tqdm(models, disable=disable_tqdm):
    weights[:, model['subset']] = model['weights']

del models

combined_model = {
    'weights': np.asmatrix(weights),
    '-B': bias,
    'threshold': 0,
}
linear.save_pipeline(model_dir, preprocessor, combined_model)
shutil.rmtree(tmp_dir)

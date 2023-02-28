import argparse
import os
import pathlib
import shlex
import subprocess

import numpy as np

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
parser.add_argument('--tmp_dir', type=str, default='tmp_grid_dir',
                    help='Temporary directory name (default: %(default)s)')
parser.add_argument('--pipeline_path', type=str, default='./linear_pipeline.pickle',
                    help='Path to resulting pipeline (default: %(default)s)')

args, passthrough_args = parser.parse_known_args()
hosts = args.hosts
tmp_dir = args.tmp_dir
subdivision = args.subdivision
pipeline_path = args.pipeline_path

div = len(hosts) * subdivision
cmd = f'python main.py {" ".join(map(lambda x: shlex.quote(x), passthrough_args))}'
jobs = [
    f'{cmd} --label_subrange {i/div} {(i+1)/div} --result_dir {tmp_dir}'
    for i in range(div)
]

grid = Grid(hosts, jobs)
grid.go()

cwd = os.getcwd()
home = os.path.expanduser('~')
if cwd.startswith(home):
    cwd = cwd[len(home):]
    if cwd.startswith(os.sep):
        cwd = cwd[1:]

pathlib.Path(f'{cwd}/{tmp_dir}').mkdir(parents=True, exist_ok=True)
for host in hosts:
    subprocess.call(f'scp -qr "{host}:{cwd}/{tmp_dir}" "{tmp_dir}/{host}"',
                    shell=True, executable='/bin/bash')
    subprocess.call(f'ssh {host} "rm -r \\"{cwd}/{tmp_dir}\\""')

num_labels = 0
models = []
for root, _, files in os.walk(tmp_dir):
    for file in files:
        if file != 'linear_pipeline.pickle':
            continue
        preprocessor, model = linear.load_pipeline(f'{root}/{file}')
        num_labels = num_labels + model['subset'].size
        models.append(model)

bias = models[0]['bias']
weights = np.zeros((num_labels, models[0]['weights'].shape[1]), order='F')
for model in models:
    weights[:, model['subset']] = model['weights']

del models

combined_model = {
    'weights': weights,
    'bias': bias,
    'threshold': 0,
}
linear.save_pipeline(pipeline_path, preprocessor, combined_model)
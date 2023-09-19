# LibMultiLabel — a Library for Multi-class and Multi-label Text Classification

LibMultiLabel is a library for binary, multi-class, and multi-label classification. It has the following functionalities

- end-to-end services from raw texts to final evaluation/analysis
- support for common neural network architectures and linear classifiers
- easy hyper-parameter selection

This is an on-going development so many improvements are still being made. Comments are very welcome.

## Environments
- Python: 3.8+
- CUDA: 11.6 (if training neural networks by GPU)
- Pytorch 1.13.1+

If you have a different version of CUDA, follow the installation instructions for PyTorch LTS at their [website](https://pytorch.org/).

## Documentation
See the documentation here: https://www.csie.ntu.edu.tw/~cjlin/libmultilabel

# Distributing 1vsrest
## Installing
On each host, run under the same directory relative to home
```bash
cd ~/<same-path-for-all-hosts>
git clone -b distribute https://github.com/ntumlgroup/LibMultiLabel
cd LibMultiLabel
pip install --user --force -r requirements.txt
```
`--force` is required because we wish to override system packages.

## Setup for NFS
Additionally, two temporary directories *for each host* must exist. If LibMultiLabel resides on NFS, then run
```bash
mkdir -p <non-nfs-path>/tmp
ln -s <non-nfs-path>/tmp
mkdir -p <non-nfs-path>/.sshutils
ln -s <non-nfs-path>/.sshutils ~/.sshutils
```
where `<non-nfs-path>` is some path to a non-nfs directory, e.g. `/tmp2/$USER`.

## Setup SSH
Every host must be reachable through ssh without passwords, i.e. uses key-pairs. See `ssh-key-gen` and `ssh-copy-id`.

## Run
```bash
python distribute.py --hosts [hosts...] [args...]
```
where `hosts` are the ssh host names to distribute on. Non-distribute `args` will be passed through to `main.py`.

See `python distribute.py -h` for more details.

### Example
```bash
python distribute.py --hosts host1 host2 --linear --training_file data/rcv1/train.txt --liblinear_options="-s 2"
```
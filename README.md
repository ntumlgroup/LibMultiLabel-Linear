# LibMultiLabel — a Library for Multi-class and Multi-label Text Classification

LibMultiLabel is a library for binary, multi-class, and multi-label classification. It has the following functionalities

- end-to-end services from raw texts to final evaluation/analysis
- support for common neural network architectures and linear classifiers
- easy hyper-parameter selection

This is an on-going development so many improvements are still being made. Comments are very welcome.

## Environments
- Python: 3.9+
- CUDA: 11.6 (if training neural networks by GPU)
- Pytorch 1.13.1+

If you have a different version of CUDA, follow the installation instructions for PyTorch LTS at their [website](https://pytorch.org/).

## Extra Requirements
```bash
git clone https://github.com/ntumlgroup/blinkless
pip install blinkless
```

Currently there may be issues with file handles. If you run into an error about too many file handles, do
```bash
ulimit -Sn $(ulimit -Hn)
```

## Documentation
See the documentation here: https://www.csie.ntu.edu.tw/~cjlin/libmultilabel

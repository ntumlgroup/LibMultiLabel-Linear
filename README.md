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

## Weight Pruning Experiment

The goal of this experiment is to reduce memory usage of libmultilabel models by exploring weights pruning across model weights with various pruning methods. The pruning-threshold experiments utilized through this experiment are:
- 0-100% Quantile Pruning w/ 1% step size
- Geometric Series Quantile Pruning w/ specified parameters

 ## Procedure
 - Setup environment suited to run LibMultiLabel(https://www.csie.ntu.edu.tw/~cjlin/libmultilabel/)
 - Modify experiment_config.yaml to parameters to fit experiment
 - Run experiment_run.sh
# tree-pruning-results

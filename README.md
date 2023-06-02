# How Does Information Bottleneck Help Deep Learning?

This repository contains code for the ICML 2023 paper: [How Does Information Bottleneck Help Deep Learning?](https://arxiv.org/abs/2305.18887)

<p align="center">
    <img src="toy/cartoon.PNG" width=50% height=50%>
</p>

## Requirements

The code used Python 3.6.9 and PyTorch 1.10.1. A full list of environmental dependencies is given in `requirements.txt`.

## Running

The experiments for each dataset follow the same procedure. First the models are trained and saved along with SWAG parameters and initial metrics, then model compression metrics and results are computed.

The scripts for running experiments are given by:

- Clustering: `toy/scripts/main_swag.py`
- MNIST: `toy/scripts/main_swag_MNIST.py`
- CIFAR: `dnn/swag_repo/MI/main_multiseed.py`
- Binning: `toy/scripts/binning/main_swag_bin.py`

## Citation
```
@inproceedings{kenji2023infodl,
  title={How Does Information Bottleneck Help Deep Learning?},
  author={Kenji Kawaguchi and Zhun Deng and Xu Ji and Jiaoyang Huang},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2023}
}
```

## Acknowledgements

The SWAG part of the code is adapted from [A Simple Baseline for Bayesian Uncertainty
in Deep Learning](https://github.com/wjmaddox/swa_gaussian).

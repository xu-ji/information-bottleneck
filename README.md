# How Does Information Bottleneck Help Deep Learning?

This repository contains code for `How Does Information Bottleneck Help Deep Learning?`.

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

TBC

## Acknowledgements

The SWAG part of the code is adapted from [A Simple Baseline for Bayesian Uncertainty
in Deep Learning](https://github.com/wjmaddox/swa_gaussian).
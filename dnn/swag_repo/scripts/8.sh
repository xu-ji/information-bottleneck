#!/bin/bash

export CUDA_VISIBLE_DEVICES=0 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_4 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 64 --wd=0.0001 --model=PreResNet83 > logs/job_112.txt

export CUDA_VISIBLE_DEVICES=0 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_4 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 64 --wd=0.0001 --model=PreResNet110 > logs/job_113.txt

export CUDA_VISIBLE_DEVICES=0 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_4 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 64 --wd=0.001 --model=PreResNet56 > logs/job_114.txt

export CUDA_VISIBLE_DEVICES=0 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_4 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 64 --wd=0.001 --model=PreResNet83 > logs/job_115.txt

export CUDA_VISIBLE_DEVICES=0 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_4 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 64 --wd=0.001 --model=PreResNet110 > logs/job_116.txt

export CUDA_VISIBLE_DEVICES=0 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_4 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 128 --wd=1e-05 --model=PreResNet56 > logs/job_117.txt

export CUDA_VISIBLE_DEVICES=0 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_4 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 128 --wd=1e-05 --model=PreResNet83 > logs/job_118.txt

export CUDA_VISIBLE_DEVICES=0 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_4 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 128 --wd=1e-05 --model=PreResNet110 > logs/job_119.txt

export CUDA_VISIBLE_DEVICES=0 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_4 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 128 --wd=0.0001 --model=PreResNet56 > logs/job_120.txt

export CUDA_VISIBLE_DEVICES=0 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_4 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 128 --wd=0.0001 --model=PreResNet83 > logs/job_121.txt

export CUDA_VISIBLE_DEVICES=0 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_4 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 128 --wd=0.0001 --model=PreResNet110 > logs/job_122.txt

export CUDA_VISIBLE_DEVICES=0 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_4 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 128 --wd=0.001 --model=PreResNet56 > logs/job_123.txt

export CUDA_VISIBLE_DEVICES=0 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_4 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 128 --wd=0.001 --model=PreResNet83 > logs/job_124.txt

export CUDA_VISIBLE_DEVICES=0 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_4 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 128 --wd=0.001 --model=PreResNet110 > logs/job_125.txt


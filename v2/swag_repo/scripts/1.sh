#!/bin/bash

export CUDA_VISIBLE_DEVICES=0 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_0 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 128 --wd=0.0001 --model=PreResNet110 > logs/job_14.txt

export CUDA_VISIBLE_DEVICES=0 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_0 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 128 --wd=0.001 --model=PreResNet56 > logs/job_15.txt

export CUDA_VISIBLE_DEVICES=0 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_0 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 128 --wd=0.001 --model=PreResNet83 > logs/job_16.txt

export CUDA_VISIBLE_DEVICES=0 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_0 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 128 --wd=0.001 --model=PreResNet110 > logs/job_17.txt

export CUDA_VISIBLE_DEVICES=0 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_0 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 1024 --wd=1e-05 --model=PreResNet56 > logs/job_18.txt

export CUDA_VISIBLE_DEVICES=0 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_0 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 1024 --wd=1e-05 --model=PreResNet83 > logs/job_19.txt

export CUDA_VISIBLE_DEVICES=0 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_0 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 1024 --wd=1e-05 --model=PreResNet110 > logs/job_20.txt

export CUDA_VISIBLE_DEVICES=0 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_0 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 1024 --wd=0.0001 --model=PreResNet56 > logs/job_21.txt

export CUDA_VISIBLE_DEVICES=0 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_0 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 1024 --wd=0.0001 --model=PreResNet83 > logs/job_22.txt

export CUDA_VISIBLE_DEVICES=0 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_0 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 1024 --wd=0.0001 --model=PreResNet110 > logs/job_23.txt

export CUDA_VISIBLE_DEVICES=0 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_0 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 1024 --wd=0.001 --model=PreResNet56 > logs/job_24.txt

export CUDA_VISIBLE_DEVICES=0 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_0 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 1024 --wd=0.001 --model=PreResNet83 > logs/job_25.txt

export CUDA_VISIBLE_DEVICES=0 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_0 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 1024 --wd=0.001 --model=PreResNet110 > logs/job_26.txt

export CUDA_VISIBLE_DEVICES=0 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_1 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 64 --wd=1e-05 --model=PreResNet56 > logs/job_27.txt


#!/bin/bash

export CUDA_VISIBLE_DEVICES=1 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_2 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 128 --wd=0.001 --model=PreResNet83 > logs/job_70.txt

export CUDA_VISIBLE_DEVICES=1 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_2 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 128 --wd=0.001 --model=PreResNet110 > logs/job_71.txt

export CUDA_VISIBLE_DEVICES=1 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_2 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 1024 --wd=1e-05 --model=PreResNet56 > logs/job_72.txt

export CUDA_VISIBLE_DEVICES=1 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_2 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 1024 --wd=1e-05 --model=PreResNet83 > logs/job_73.txt

export CUDA_VISIBLE_DEVICES=1 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_2 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 1024 --wd=1e-05 --model=PreResNet110 > logs/job_74.txt

export CUDA_VISIBLE_DEVICES=1 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_2 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 1024 --wd=0.0001 --model=PreResNet56 > logs/job_75.txt

export CUDA_VISIBLE_DEVICES=1 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_2 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 1024 --wd=0.0001 --model=PreResNet83 > logs/job_76.txt

export CUDA_VISIBLE_DEVICES=1 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_2 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 1024 --wd=0.0001 --model=PreResNet110 > logs/job_77.txt

export CUDA_VISIBLE_DEVICES=1 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_2 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 1024 --wd=0.001 --model=PreResNet56 > logs/job_78.txt

export CUDA_VISIBLE_DEVICES=1 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_2 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 1024 --wd=0.001 --model=PreResNet83 > logs/job_79.txt

export CUDA_VISIBLE_DEVICES=1 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_2 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 1024 --wd=0.001 --model=PreResNet110 > logs/job_80.txt

export CUDA_VISIBLE_DEVICES=1 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_3 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 64 --wd=1e-05 --model=PreResNet56 > logs/job_81.txt

export CUDA_VISIBLE_DEVICES=1 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_3 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 64 --wd=1e-05 --model=PreResNet83 > logs/job_82.txt

export CUDA_VISIBLE_DEVICES=1 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_3 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 64 --wd=1e-05 --model=PreResNet110 > logs/job_83.txt


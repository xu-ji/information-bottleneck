#!/bin/bash

export CUDA_VISIBLE_DEVICES=0 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_0 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 64 --wd=1e-05 --model=PreResNet56 > logs/job_0.txt

export CUDA_VISIBLE_DEVICES=0 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_0 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 64 --wd=1e-05 --model=PreResNet83 > logs/job_1.txt

export CUDA_VISIBLE_DEVICES=0 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_0 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 64 --wd=1e-05 --model=PreResNet110 > logs/job_2.txt

export CUDA_VISIBLE_DEVICES=0 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_0 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 64 --wd=0.0001 --model=PreResNet56 > logs/job_3.txt

export CUDA_VISIBLE_DEVICES=0 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_0 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 64 --wd=0.0001 --model=PreResNet83 > logs/job_4.txt

export CUDA_VISIBLE_DEVICES=0 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_0 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 64 --wd=0.0001 --model=PreResNet110 > logs/job_5.txt

export CUDA_VISIBLE_DEVICES=0 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_0 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 64 --wd=0.001 --model=PreResNet56 > logs/job_6.txt

export CUDA_VISIBLE_DEVICES=0 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_0 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 64 --wd=0.001 --model=PreResNet83 > logs/job_7.txt

export CUDA_VISIBLE_DEVICES=0 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_0 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 64 --wd=0.001 --model=PreResNet110 > logs/job_8.txt

export CUDA_VISIBLE_DEVICES=0 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_0 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 128 --wd=1e-05 --model=PreResNet56 > logs/job_9.txt

export CUDA_VISIBLE_DEVICES=0 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_0 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 128 --wd=1e-05 --model=PreResNet83 > logs/job_10.txt

export CUDA_VISIBLE_DEVICES=0 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_0 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 128 --wd=1e-05 --model=PreResNet110 > logs/job_11.txt

export CUDA_VISIBLE_DEVICES=0 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_0 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 128 --wd=0.0001 --model=PreResNet56 > logs/job_12.txt

export CUDA_VISIBLE_DEVICES=0 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_0 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 128 --wd=0.0001 --model=PreResNet83 > logs/job_13.txt


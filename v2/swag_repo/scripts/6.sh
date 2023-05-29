#!/bin/bash

export CUDA_VISIBLE_DEVICES=0 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_3 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 64 --wd=0.0001 --model=PreResNet56 > logs/job_84.txt

export CUDA_VISIBLE_DEVICES=0 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_3 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 64 --wd=0.0001 --model=PreResNet83 > logs/job_85.txt

export CUDA_VISIBLE_DEVICES=0 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_3 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 64 --wd=0.0001 --model=PreResNet110 > logs/job_86.txt

export CUDA_VISIBLE_DEVICES=0 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_3 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 64 --wd=0.001 --model=PreResNet56 > logs/job_87.txt

export CUDA_VISIBLE_DEVICES=0 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_3 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 64 --wd=0.001 --model=PreResNet83 > logs/job_88.txt

export CUDA_VISIBLE_DEVICES=0 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_3 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 64 --wd=0.001 --model=PreResNet110 > logs/job_89.txt

export CUDA_VISIBLE_DEVICES=0 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_3 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 128 --wd=1e-05 --model=PreResNet56 > logs/job_90.txt

export CUDA_VISIBLE_DEVICES=0 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_3 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 128 --wd=1e-05 --model=PreResNet83 > logs/job_91.txt

export CUDA_VISIBLE_DEVICES=0 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_3 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 128 --wd=1e-05 --model=PreResNet110 > logs/job_92.txt

export CUDA_VISIBLE_DEVICES=0 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_3 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 128 --wd=0.0001 --model=PreResNet56 > logs/job_93.txt

export CUDA_VISIBLE_DEVICES=0 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_3 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 128 --wd=0.0001 --model=PreResNet83 > logs/job_94.txt

export CUDA_VISIBLE_DEVICES=0 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_3 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 128 --wd=0.0001 --model=PreResNet110 > logs/job_95.txt

export CUDA_VISIBLE_DEVICES=0 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_3 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 128 --wd=0.001 --model=PreResNet56 > logs/job_96.txt

export CUDA_VISIBLE_DEVICES=0 && python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 --seed 0 --dataset=CIFAR10_seed_0_inst_3 --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 --epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size 128 --wd=0.001 --model=PreResNet83 > logs/job_97.txt


import numpy as np
import os

seeds = [0, 1, 2, 3]

dataset_instances = list(range(5))

batch_sizes = [64, 128, 1024]
decays = [1e-5, 1e-4, 1e-3]
model_names = ["PreResNet56", "PreResNet83", "PreResNet110"]

commands_per_script = 12

# generate commands

template = "python -m experiments.train.run_swag  --dir=/network/scratch/x/xu.ji/mutual_info/v2 " \
           "--seed %d --dataset=CIFAR10_seed_%d_inst_%s --data_path=/network/scratch/x/xu.ji/datasets/CIFAR10 " \
           "--epochs=200 --lr_init=0.1 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test --batch_size %s " \
           "--wd=%s --model=%s > logs/job_%s.txt"

command_ind = 0

commands = []
for seed in seeds:
    for d_i in dataset_instances:
        for b in batch_sizes:
            for d in decays:
                for m in model_names:
                    command = template % (seed, seed, d_i, b, d, m, command_ind) #
                    command_ind += 1
                    commands.append(command)

# put commands into bash scripts, prepending device
num_scripts = int(np.ceil(command_ind / float(commands_per_script)))

print("Num commands and scripts")
print(command_ind)
print(num_scripts)


os.makedirs("scripts_gen", exist_ok=True)
curr_ind = 0
for s in range(num_scripts):
    with open("scripts_gen/%s.sh" % (s), "w") as f:
        f.write("#!/bin/bash")
        f.write("\n")
        f.write("\n")

        for c in range(curr_ind, min(curr_ind + commands_per_script, command_ind)):
            f.write(commands[c])
            f.write("\n")
            f.write("\n")

            curr_ind += 1

print("Don't forget to chmod +x !")

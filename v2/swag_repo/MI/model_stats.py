import os

import torch

num_samples = 5 # per swag model in the posterior - ie per num dataset * num seed
data_instances = list(range(5))
num_layers = 5
num_seeds = 4

sz1 = 2000
sz1_y = 1000
sz2 = 400
batch_sz = 70

batch_sizes = [64, 128, 1024]
decays = [1e-5, 0.0001, 0.001] # weird formatting
model_names = ["PreResNet56", "PreResNet83", "PreResNet110"]

axis_items = [
    batch_sizes,
    decays,
    model_names
]

settings = []
for b in batch_sizes:
    for d in decays:
        for m in model_names:
            settings.append((b, d, m))

train_err = []
train_loss = []
test_err = []
test_loss = []
performance_out_dir = os.path.join("/network/scratch/x/xu.ji/mutual_info/v7/" "performance")
performance_compute_ind = 0
for seed in range(num_seeds):
    for s_i, (batch_size, wd, model_name) in enumerate(settings):
        for d_i in data_instances:
            performance_path = os.path.join(performance_out_dir,
                                            "performance_%s.pt" % performance_compute_ind)

            performance = torch.load(performance_path)

            train_err.append(performance["train_err"])
            train_loss.append(performance["train_loss"])
            test_err.append(performance["test_err"])
            test_loss.append(performance["test_loss"])

            performance_compute_ind += 1

train_accs = 1 - torch.tensor(train_err)
train_losses = torch.tensor(train_loss)
test_accs = 1 - torch.tensor(test_err)
test_losses = torch.tensor(test_loss)

print("Num models")
print(train_accs.shape)

print("Train loss & %.4f & %.4f & %.4f & %.4f \\\\" % (train_losses.mean(), train_losses.std(), train_losses.max(), train_losses.min()))
print("Train accuracy & %.4f & %.4f & %.4f & %.4f \\\\" % (train_accs.mean(), train_accs.std(), train_accs.max(), train_accs.min()))

print("Test loss & %.4f & %.4f & %.4f & %.4f \\\\" % (test_losses.mean(), test_losses.std(), test_losses.max(), test_losses.min()))
print("Test accuracy & %.4f & %.4f & %.4f & %.4f \\\\" % (test_accs.mean(), test_accs.std(), test_accs.max(), test_accs.min()))

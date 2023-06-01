import os
import torch
from collections import defaultdict

# printing train, test statistics for accepted models

# from main_swag

out_dir = "/network/scratch/x/xu.ji/mutual_info/v6c"
print(out_dir)

data_instances = list(range(3))

archs = [[2, 256, 256, 128, 128, 5],
[2, 128, 128, 64, 64, 5],
[2, 64, 64, 32, 32, 5],
[2, 32, 32, 16, 16, 5],
]

decays = [0.0, 1e-2, 1e-1]

lamb_lrs = [0.0, 5e-4, 1e-4]

seeds = [0, 1, 2]

train_accs = []
test_accs = []
train_losses = []
test_losses = []

MIs = defaultdict(list)
MIs_5 = defaultdict(list)
MIs_10 = defaultdict(list)

neggap = []
neggapacc = []

gengap = []
gengapacc = []

thresh = 0.85
skipped = 0
considered = 0

model_ind = 0
for lamb_lr in lamb_lrs:
    for arch in archs:
        for decay in decays:

            for data_instance in data_instances:
                for seed in seeds:

                    if lamb_lr == 0:

                        r = torch.load(os.path.join(out_dir, "results_%d.pt" % model_ind))

                        if True: #(r["train_acc"] > thresh):
                            train_accs.append(r["train_acc"])
                            test_accs.append(r["test_acc"])

                            train_losses.append(r["train_loss"])
                            test_losses.append(r["test_loss"])

                            MIs[str(lamb_lr)].append(r["diagnostics"]["MI_mc"])
                            MIs_5[str(lamb_lr)].append(r["MI_mc_5"])
                            MIs_10[str(lamb_lr)].append(r["MI_mc_10"])

                            gengap.append(r["test_loss"] - r["train_loss"])
                            gengapacc.append(r["train_acc"] - r["test_acc"])

                            if r["test_loss"] < r["train_loss"]:
                                neggap.append(r["test_loss"] - r["train_loss"])
                                neggapacc.append(r["train_acc"] - r["test_acc"])
                        else:
                            skipped += 1

                        considered += 1

                    model_ind += 1

train_accs = torch.tensor(train_accs)
test_accs = torch.tensor(test_accs)
train_losses = torch.tensor(train_losses)
test_losses = torch.tensor(test_losses)

neggap = torch.tensor(neggap)
neggapacc = torch.tensor(neggapacc)

gengap = torch.tensor(gengap)
gengapacc = torch.tensor(gengapacc)

print("All:")
print(model_ind)

print("Considered:")
print(considered)

print("Skipped:")
print(skipped)

print("Resulting:")
print(train_accs.shape)

print("Train loss & %.4f & %.4f & %.4f & %.4f \\\\" % (train_losses.max(), train_losses.min(), train_losses.mean(), train_losses.std() ))
print("Train accuracy & %.4f & %.4f & %.4f & %.4f \\\\" % (train_accs.max(), train_accs.min(), train_accs.mean(), train_accs.std()))

print("Test loss & %.4f & %.4f & %.4f & %.4f \\\\" % (test_losses.max(), test_losses.min(), test_losses.mean(), test_losses.std() ))
print("Test accuracy & %.4f & %.4f & %.4f & %.4f \\\\" % (test_accs.max(), test_accs.min(), test_accs.mean(), test_accs.std() ))

print("Neg gap %s & %.4f & %.4f & %.4f & %.4f \\\\" % (str(neggap.shape), neggap.max(), neggap.min(), neggap.mean(), neggap.std() ))
print("Neg gap acc %s & %.4f & %.4f & %.4f & %.4f \\\\" % (str(neggapacc.shape), neggapacc.max(), neggapacc.min(), neggapacc.mean(), neggapacc.std() ))

print("Gen gap %s & %.4f & %.4f & %.4f & %.4f \\\\" % (str(gengap.shape), gengap.max(), gengap.min(), gengap.mean(), gengap.std() ))
print("Gen gap acc %s & %.4f & %.4f & %.4f & %.4f \\\\" % (str(gengapacc.shape), gengapacc.max(), gengapacc.min(), gengapacc.mean(), gengapacc.std() ))


for lamb_lr in lamb_lrs:
    print(lamb_lr)
    MIs_lamb = torch.tensor(MIs[str(lamb_lr)])
    MIs_5_lamb = torch.tensor(MIs_5[str(lamb_lr)])
    MIs_10_lamb = torch.tensor(MIs_10[str(lamb_lr)])

    print("MI & %.4f & %.4f & %.4f & %.4f \\\\" % (MIs_lamb.max(), MIs_lamb.min(), MIs_lamb.mean(), MIs_lamb.std() ))

    print("MI 5 & %.4f & %.4f & %.4f & %.4f \\\\" % (MIs_5_lamb.max(), MIs_5_lamb.min(), MIs_5_lamb.mean(), MIs_5_lamb.std() ))

    print("MI 10 & %.4f & %.4f & %.4f & %.4f \\\\" % (MIs_10_lamb.max(), MIs_10_lamb.min(), MIs_10_lamb.mean(), MIs_10_lamb.std() ))
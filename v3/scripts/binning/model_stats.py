import os
import torch
from collections import defaultdict

# printing train, test statistics for accepted models

# from main_swag

out_dir = "/network/scratch/x/xu.ji/mutual_info/v33"
print(out_dir)


archs = [[2, 256, 256, 128, 128, 5],
[2, 128, 128, 64, 64, 5],
[2, 64, 64, 32, 32, 5],
[2, 32, 32, 16, 16, 5],
]

data_instances = list(range(3))

decays = [0.0, 1e-3, 1e-2]

lamb_lrs = [0.0, 1.0, 2.0, 4.0, 8.0]

seeds = [0, 1, 2]

train_accs = []
test_accs = []
train_losses = []
test_losses = []

MIs = defaultdict(list)
MIs_5 = defaultdict(list)
MIs_10 = defaultdict(list)
train_accs_lamb = defaultdict(list)
test_accs_lamb = defaultdict(list)

neggap = []
neggapacc = []

#thresh = 0.85
skipped = 0
considered = 0

train_accs_accept = []
train_loss_accept = []
train_accs_skip = []
train_loss_skip = []
test_accs_accept = []
test_loss_accept = []
test_accs_skip = []
test_loss_skip = []


model_ind = 0
for lamb_lr in lamb_lrs:
    for arch in archs:
        for decay in decays:

            for data_instance in data_instances:
                for seed in seeds:

                    try:
                        r = torch.load(os.path.join(out_dir, "results_%d.pt" % model_ind))

                        if True: #r["test_loss"] - r["train_loss"] < 0.1:
                            train_accs.append(r["train_acc"])
                            test_accs.append(r["test_acc"])

                            train_losses.append(r["train_loss"])
                            test_losses.append(r["test_loss"])

                            train_accs_lamb[str(lamb_lr)].append(r["train_acc"])
                            test_accs_lamb[str(lamb_lr)].append(r["test_acc"])

                            MIs[str(lamb_lr)].append(r["diagnostics"]["MI_mc_perlayer"][-1])
                            MIs_5[str(lamb_lr)].append(r["MI_mc_5_perlayer"][-1])
                            MIs_10[str(lamb_lr)].append(r["MI_mc_10_perlayer"][-1])

                            if r["test_loss"] < r["train_loss"]:
                                neggap.append(r["test_loss"] - r["train_loss"])
                                neggapacc.append(r["train_acc"] - r["test_acc"])

                            #print("Accepting %s" % str((arch, decay)))

                            #print("Accepting %s" % str((arch, decay, data_instance, seed)))

                            train_accs_accept.append(r["train_acc"])
                            train_loss_accept.append(r["train_loss"])
                            test_accs_accept.append(r["test_acc"])
                            test_loss_accept.append(r["test_loss"])
                        else:
                            #print("... skipping %s" % str((arch, decay, data_instance, seed)))
                            skipped += 1

                            train_accs_skip.append(r["train_acc"])
                            train_loss_skip.append(r["train_loss"])
                            test_accs_skip.append(r["test_acc"])
                            test_loss_skip.append(r["test_loss"])
                        considered += 1
                    except:
                        pass

                    model_ind += 1

train_accs = torch.tensor(train_accs)
test_accs = torch.tensor(test_accs)
train_losses = torch.tensor(train_losses)
test_losses = torch.tensor(test_losses)

neggap = torch.tensor(neggap)
neggapacc = torch.tensor(neggapacc)

train_accs_accept = torch.tensor(train_accs_accept)
train_loss_accept = torch.tensor(train_loss_accept)
train_accs_skip = torch.tensor(train_accs_skip)
train_loss_skip = torch.tensor(train_loss_skip)

test_accs_accept = torch.tensor(test_accs_accept)
test_loss_accept = torch.tensor(test_loss_accept)
test_accs_skip = torch.tensor(test_accs_skip)
test_loss_skip = torch.tensor(test_loss_skip)


print("Thresh details")
print((train_accs_accept.mean().item(), train_accs_accept.max().item(), train_accs_accept.min().item()))
print((train_loss_accept.mean().item(), train_loss_accept.max().item(), train_loss_accept.min().item()))
print((test_accs_accept.mean().item(), test_accs_accept.max().item(), test_accs_accept.min().item()))
print((test_loss_accept.mean().item(), test_loss_accept.max().item(), test_loss_accept.min().item()))
print("--")
if train_accs_skip.shape[0] > 0:
    print((train_accs_skip.mean().item(), train_accs_skip.max().item(), train_accs_skip.min().item()))
    print((train_loss_skip.mean().item(), train_loss_skip.max().item(), train_loss_skip.min().item()))
    print((test_accs_skip.mean().item(), test_accs_skip.max().item(), test_accs_skip.min().item()))
    print((test_loss_skip.mean().item(), test_loss_skip.max().item(), test_loss_skip.min().item()))


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


print("---")

for lamb_lr in lamb_lrs:
    print(lamb_lr)
    MIs_lamb = torch.tensor(MIs[str(lamb_lr)])
    MIs_5_lamb = torch.tensor(MIs_5[str(lamb_lr)])
    MIs_10_lamb = torch.tensor(MIs_10[str(lamb_lr)])

    train_accs_lamb_curr = torch.tensor(train_accs_lamb[str(lamb_lr)])
    test_accs_lamb_curr = torch.tensor(test_accs_lamb[str(lamb_lr)])

    print("train acc & %.4f & %.4f & %.4f & %.4f \\\\" % (train_accs_lamb_curr.max(), train_accs_lamb_curr.min(), train_accs_lamb_curr.mean(), train_accs_lamb_curr.std() ))

    print("test acc & %.4f & %.4f & %.4f & %.4f \\\\" % (test_accs_lamb_curr.max(), test_accs_lamb_curr.min(), test_accs_lamb_curr.mean(), test_accs_lamb_curr.std() ))

    print("MI & %.4f & %.4f & %.4f & %.4f \\\\" % (MIs_lamb.max(), MIs_lamb.min(), MIs_lamb.mean(), MIs_lamb.std() ))

    print("MI 5 & %.4f & %.4f & %.4f & %.4f \\\\" % (MIs_5_lamb.max(), MIs_5_lamb.min(), MIs_5_lamb.mean(), MIs_5_lamb.std() ))

    print("MI 10 & %.4f & %.4f & %.4f & %.4f \\\\" % (MIs_10_lamb.max(), MIs_10_lamb.min(), MIs_10_lamb.mean(), MIs_10_lamb.std() ))
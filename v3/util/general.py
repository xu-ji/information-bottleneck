
import random
import numpy as np
import torch
from torch import nn
import gc
from collections import defaultdict
from collections import OrderedDict
import sys
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

device = torch.device("cuda:0")

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


def train_model(model, opt, train_data, test_data, args, lamb_lr):
    model.train()

    train_accs, train_losses = [], []
    test_accs, test_losses = [], []
    l_MIs = []
    maxes = []
    lambs = []

    t = 0
    lamb = args.lamb_init

    analyse(model, grads=True)

    for ep in range(args.epochs):
        for xs, ys in train_data:

            if t % 10 == 0:
                train_acc, train_loss = evaluate(model, train_data, args, "train", plot=False)
                test_acc, test_loss = evaluate(model, test_data, args, "test", plot=False)
                train_accs.append(train_acc)
                train_losses.append(train_loss)
                test_accs.append(test_acc)
                test_losses.append(test_loss)
                sys.stdout.flush()

            xs = xs.to(device)
            ys = ys.to(device)

            opt.zero_grad()

            preds, max_prob = model(xs)
            l_sup = nn.functional.cross_entropy(preds, ys, reduction="mean")

            maxes.append(max_prob)

            MI = est_MI(model, train_data.dataset, sz=min(100, len(train_data.dataset)))
            # when this is -ve, lambda becomes smaller. When positive, lambda becomes bigger
            constraint = (args.MI_const - MI)

            l_MIs.append(MI.item())

            loss = l_sup + lamb * constraint

            #print(("training loop", l_sup.item(), MI.item(), constraint.item(), lamb))

            loss.backward()

            opt.step()

            analyse(model, grads=True, t=t)

            lamb += lamb_lr * constraint.item() # gradient ascent
            lambs.append(lamb)

            t += 1

    MI = est_MI(model, train_data.dataset, sz=min(1000, len(train_data.dataset)), requires_grad=False)

    diagnostics = {"train_losses": train_losses,
                   "train_accs": train_accs,
                   "test_losses": test_losses,
                   "test_accs": test_accs,
                   "l_MIs": l_MIs,
                   "maxes": maxes,
                   "lambs": lambs,
                   }
    return model, MI.item(), diagnostics


def est_MI(model, dataset, sz, jensen, requires_grad=True):
    ii = np.random.choice(len(dataset), size=sz, replace=False)
    x = torch.stack([dataset[i][0] for i in ii], dim=0).to(device)

    if not requires_grad:
        model.eval()
        with torch.no_grad():
            z, log_prob = model(x, repr=True)
            log_marg_prob = model.log_marg_prob(z, x, jensen=jensen)
        model.train()
    else:
        z, log_prob = model(x, repr=True)
        log_marg_prob = model.log_marg_prob(z, x, jensen=jensen)

    return (log_prob - log_marg_prob).mean()



def est_MI_cond(model, num_classes, dl, sz, jensen): # todo was changed to require size
    model.eval()

    x_class = [[] for _ in range(num_classes)]
    counts = torch.zeros(num_classes, device=device)

    for xs, ys in dl:
        xs = xs.to(device)
        ys = ys.to(device)

        for c in range(num_classes):
            c_inds = ys == c
            x_class[c].append(xs[c_inds])
            counts[c] += c_inds.sum()

    MIs = torch.zeros(num_classes, device=device)
    for c in range(num_classes):
        #x_class_c = torch.cat(x_class[c], dim=0)
        x_class_c = torch.cat(x_class[c], dim=0).to(device)

        if sz != -1:
            ii = np.random.choice(len(x_class[c]), size=sz, replace=False)
            x_class_c = torch.stack([x_class_c[i] for i in ii], dim=0).to(device)

        with torch.no_grad():
            z, log_prob = model(x_class_c, repr=True)
            log_marg_prob = model.log_marg_prob(z, x_class_c, jensen=jensen)
            MIs[c] = (log_prob - log_marg_prob).mean()

    counts = counts / counts.sum()
    assert (counts.shape == MIs.shape)
    MI_avg = (MIs * counts).sum()

    model.train()
    return MI_avg


def compute_factors(model, dl, num_classes, jensen):
    # just use all of training set as typical set for now
    model.eval()

    x_class = [[] for _ in range(num_classes)]
    counts = torch.zeros(num_classes, device=device)

    for xs, ys in dl:
        xs = xs.to(device)
        ys = ys.to(device)

        for c in range(num_classes):
            c_inds = ys == c
            x_class[c].append(xs[c_inds])
            counts[c] += c_inds.sum()

    GL3_loss, GL3_err = [], []
    for c in range(num_classes):
        x_class_c = torch.cat(x_class[c], dim=0).to(device)

        with torch.no_grad():
            z, _ = model(x_class_c, repr=True)
            log_marg_prob = model.log_marg_prob(z, x_class_c, jensen=jensen) # one per item
            assert log_marg_prob.shape == (x_class_c.shape[0],)
            marg_prob = torch.exp(log_marg_prob)

            preds, _ = model(x_class_c)
            l_sups = nn.functional.cross_entropy(preds, torch.ones(x_class_c.shape[0], dtype=torch.long, device=device).fill_(c), reduction="none")
            err_sups = (preds.argmax(dim=1) != c).to(torch.float)

            assert l_sups.shape == (x_class_c.shape[0],) and err_sups.shape == (x_class_c.shape[0],)

        GL3_loss_c = (torch.sqrt(2. * num_classes * marg_prob) * l_sups).sum().item()
        GL3_err_c = (torch.sqrt(2. * num_classes * marg_prob) * err_sups).sum().item()

        GL3_loss.append(GL3_loss_c)
        GL3_err.append(GL3_err_c)

    return {"GL3_loss": GL3_loss, "GL3_err": GL3_err}


def compute_factors_binning(model, dataset, num_bins, num_classes):
    x = torch.stack([dataset[i][0] for i in range(len(dataset))], dim=0).to(device)
    y = torch.tensor([dataset[i][1] for i in range(len(dataset))]).to(device)
    assert len(y.shape) == 1
    #print("factors shape")
    #print((x.shape, y.shape))
    sys.stdout.flush()

    model.eval()  # turn off dropout

    with torch.no_grad():
        zs = model(x, repr=True)
    model.train()

    ps = []
    for feats in zs:
        max_val = feats.max().item()  # or per cell? no
        min_val = feats.min().item()
        binsize = (max_val - min_val) / num_bins

        feats_shifted = feats - min_val
        feats_np = feats_shifted.cpu().numpy()

        #h_z = get_h(feats_np, binsize)
        digitized = np.floor(feats_np / binsize).astype('int')
        p_ts_unique, inverses = get_unique_probs(digitized) # per unique pattern!

        p_ts = p_ts_unique[inverses]
        ps.append(torch.tensor(p_ts))

    ps = torch.stack(ps, dim=1).to(device) # num inputs, num layers
    assert ps.shape == (x.shape[0], len(zs))

    GL3_loss, GL3_err = [], []
    for c in range(num_classes):
        c_inds = (y == c)
        new_num = 3 # smaller typical set
        #print("orig size %s -> %s" % (c_inds.sum(), new_num))

        x_c = x[c_inds][:new_num]
        y_c = y[c_inds][:new_num]

        with torch.no_grad():
            preds, _ = model(x_c)
        l_sups = nn.functional.cross_entropy(preds, y_c, reduction="none")
        err_sups = (preds.argmax(dim=1) != c).to(torch.float)

        assert l_sups.shape == (x_c.shape[0],) and err_sups.shape == (x_c.shape[0],)

        probs_c = ps[c_inds, :][:new_num, :] # num inputs, num_layers

        GL3_loss_c = (torch.sqrt(2. * num_classes * probs_c) * l_sups.unsqueeze(dim=1)).sum(dim=0).cpu()
        GL3_err_c = (torch.sqrt(2. * num_classes * probs_c) * err_sups.unsqueeze(dim=1)).sum(dim=0).cpu()

        GL3_loss.append(GL3_loss_c) # each per num_layer
        GL3_err.append(GL3_err_c)

    return {"GL3_loss": torch.stack(GL3_loss, dim=0).numpy(), "GL3_err": torch.stack(GL3_err, dim=0).numpy()} #  num classes, num layer


# https://github.com/artemyk/ibsgd/blob/1ccf656f87418ffc108d2469fdea4ae2b493d8b7/simplebinmi.py#L4
def get_unique_probs(x):
    uniqueids = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
    _, unique_inverse, unique_counts = np.unique(uniqueids, return_index=False, return_inverse=True, return_counts=True)
    return np.asarray(unique_counts / float(sum(unique_counts))), unique_inverse


# https://github.com/artemyk/ibsgd/blob/iclr2018/simplebinmi.py
def get_h(d, binsize):
    digitized = np.floor(d / binsize).astype('int')
    p_ts, _ = get_unique_probs(digitized)
    return -np.sum(p_ts * np.log(p_ts))


def est_MI_binning(model, dataset, num_bins):
    #ii = np.random.choice(len(dataset), size=sz, replace=False)
    x = torch.stack([dataset[i][0] for i in range(len(dataset))], dim=0).to(device)

    model.eval() # turn off dropout

    with torch.no_grad():
        zs = model(x, repr=True)
    model.train()

    # H(Z | X) = 0
    # MI(Z; X) = H(Z)
    h_z_all = []
    ranges = []
    bin_szs = []
    for feats in zs:
        max_val = feats.max().item() # or per cell? no
        min_val = feats.min().item()
        binsize = (max_val - min_val) / num_bins

        feats_shifted = feats - min_val
        feats_np = feats_shifted.cpu().numpy()
        h_z = get_h(feats_np, binsize)

        h_z_all.append(h_z)
        ranges.append((min_val, max_val))
        bin_szs.append(binsize)

    return h_z_all, ranges, bin_szs


def est_MI_binning_cond(model, dataset, num_classes, num_bins_cond):
    model.eval()

    x_class = [[] for _ in range(num_classes)]

    for x, c in dataset:
        x_class[c].append((x, c))

    MIs = []
    for c in range(num_classes):
        h_z_all, _, _ = est_MI_binning(model, x_class[c], num_bins_cond)
        MIs.append(h_z_all)

    model.train()
    return MIs


def evaluate(model, data, args, s, plot=False):
    model.eval()

    accs = []
    losses = []

    all_hard = []
    all_xs = []
    with torch.no_grad():
        for xs, ys in data:
            xs = xs.to(device)
            ys = ys.to(device)

            preds, _ = model(xs)

            loss = torch.nn.functional.cross_entropy(preds, ys, reduction="none")
            losses.append(loss)

            hard = preds.argmax(dim=1)
            acc = (hard == ys).to(torch.float)
            accs.append(acc)

            all_hard.append(hard)
            all_xs.append(xs)

    if plot:
        f, ax = plt.subplots(1)

        all_xs = torch.cat(all_xs, dim=0)
        all_hard = torch.cat(all_hard, dim=0)
        for c in range(args.C):
            ax.scatter(all_xs[all_hard == c, 0].cpu().numpy(), all_xs[all_hard == c, 1].cpu().numpy())

        plt.tight_layout()
        f.savefig(os.path.join(args.out_dir, "preds_%s.png" % s), bbox_inches="tight")
        plt.close("all")

    model.train()
    return torch.cat(accs).mean().item(), torch.cat(losses).mean().item()


def analyse(model, grads=True, t=None):
    all_val = []
    all_grads = []
    all_val_m = []
    all_grads_m = []
    for p in model.parameters():
        all_val.append(p.data.abs().max().item())
        all_val_m.append(p.data.abs().mean().item())
        if grads and p.grad is not None:
            all_grads.append(p.grad.abs().max().item())
            all_grads_m.append(p.grad.abs().mean().item())

    val_m = np.array(all_val_m).mean()
    max_grad = None
    grad_m = None
    if grads and len(all_grads) > 0:
        max_grad = max(all_grads)
        grad_m = np.array(all_grads_m).mean()
    print("\t analyse %s: params max %s mean %s, grads max %s mean %s" % (
    t, max(all_val), val_m, max_grad, grad_m))
    return val_m


def clean(s):
    return s.replace(" ", "_")



def clean_rev(s):
    return s.replace("_", " ")


def get_weight_norms(model):
    norms = []
    num_params = 0
    for p in model.parameters():
        num_params += p.data.numel()
        if len(p.data.shape) == 2:
            norms.append(torch.linalg.norm(p.data, ord="fro"))
        else:
            norms.append(torch.linalg.norm(p.data.flatten(), ord=2))

    return torch.tensor(norms), num_params
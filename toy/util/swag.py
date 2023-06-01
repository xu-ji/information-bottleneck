import random
import numpy as np
import torch
from torch import nn
import gc
from collections import defaultdict
from collections import OrderedDict
import sys
import os
from toy.util.general import analyse, est_MI, est_MI_cond, evaluate, device, est_MI_binning, est_MI_binning_cond
from toy.util.model import StochasticMLP, StochasticConvMLP, BasicMLP

from dnn.swag_repo.swag.posteriors.swag import SWAG
from dnn.swag_repo.swag.utils import eval as swag_eval
#from v2.swag_repo.swag.losses import cross_entropy as swag_cross_entropy
import traceback

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MI_SZ = 256

def train_model_swag(model, arch, opt, train_data, test_data, args, lamb_lr, verbose=False):
    model.train()

    MI_data = train_data

    train_accs, train_losses = [], []
    test_accs, test_losses = [], []

    l_MIs = []
    maxes = []
    lambs = []

    t = 0
    lamb = args.lamb_init

    analyse(model, grads=True)

    if isinstance(model, StochasticMLP):
        swag_model = SWAG(
            StochasticMLP,
            no_cov_mat=False,
            max_num_models=10,
            arch=arch
        )
    elif isinstance(model, StochasticConvMLP):
        swag_model = SWAG(
            StochasticConvMLP,
            no_cov_mat=False,
            max_num_models=10,
            arch=arch,
            in_dim=args.in_dim
        )

    swag_model.to(device)

    for ep in range(args.epochs):
        if ep >= args.swa_start:
            swag_model.collect_model(model)

        for xs, ys in train_data:

            if t % 10 == 0:
                train_acc, train_loss = evaluate(model, train_data, args, "train", plot=False)
                test_acc, test_loss = evaluate(model, test_data, args, "test", plot=False)
                train_accs.append(train_acc)
                train_losses.append(train_loss)
                test_accs.append(test_acc)
                test_losses.append(test_loss)

                if verbose:
                    print((ep, t, train_acc, test_acc))
                    sys.stdout.flush()

            xs = xs.to(device)
            ys = ys.to(device)

            opt.zero_grad()

            preds, max_prob = model(xs)
            l_sup = nn.functional.cross_entropy(preds, ys, reduction="mean")

            maxes.append(max_prob)

            MI = est_MI(model, MI_data.dataset, jensen=False, sz=min(MI_SZ, len(MI_data.dataset)))
            # when this is -ve, lambda becomes smaller. When positive, lambda becomes bigger
            constraint = (args.MI_const - MI)

            l_MIs.append(MI.item())

            loss = l_sup + lamb * constraint

            loss.backward()

            opt.step()

            if ep == 0: analyse(model, grads=True, t=t)

            lamb += lamb_lr * constraint.item() # gradient ascent
            lambs.append(lamb)


            t += 1

    # Final eval
    train_acc, train_loss = evaluate(model, train_data, args, "train", plot=False)
    test_acc, test_loss = evaluate(model, test_data, args, "test", plot=False)
    train_accs.append(train_acc)
    train_losses.append(train_loss)
    test_accs.append(test_acc)
    test_losses.append(test_loss)

    MI_mc = est_MI(model, MI_data.dataset, sz=min(MI_SZ, len(MI_data.dataset)), jensen=False, requires_grad=False).item()
    MI_cond_mc = est_MI_cond(model, args.C, MI_data, sz=-1, jensen=False).item()

    MI_jensen = est_MI(model, MI_data.dataset, sz=min(MI_SZ, len(MI_data.dataset)), jensen=True, requires_grad=False).item()
    MI_cond_jensen = est_MI_cond(model, args.C, MI_data, sz=-1, jensen=True).item()

    swag_model.sample(0.0)
    test_acc_swag, test_loss_swag = evaluate(swag_model, test_data, args, "test", plot=False)

    diagnostics = {"train_losses": train_losses,
                   "train_accs": train_accs,
                   "test_losses": test_losses,
                   "test_accs": test_accs,
                   "test_loss_swag": test_loss_swag,
                   "test_acc_swag": test_acc_swag,

                   "MI_mc": MI_mc,
                   "MI_cond_mc": MI_cond_mc,
                   "MI_jensen": MI_jensen,
                   "MI_cond_jensen": MI_cond_jensen,

                   "l_MIs": l_MIs,
                   "maxes": maxes,
                   "lambs": lambs,
                   }

    return model, swag_model, diagnostics


def train_model_swag_binning(model, arch, opt, train_data, test_data, args, lamb_lr, verbose=True):
    model.train()

    MI_data = train_data

    train_accs, train_losses = [], []
    test_accs, test_losses = [], []

    binning_MIs = []
    l_MIs = []
    maxes = []

    t = 0

    analyse(model, grads=True)

    if isinstance(model, BasicMLP):
        swag_model = SWAG(
            BasicMLP,
            no_cov_mat=False,
            max_num_models=10,
            arch=arch
        )

    swag_model.to(device)

    for ep in range(args.epochs):
        if ep >= args.swa_start:
            swag_model.collect_model(model)

        for xs, ys in train_data:

            if t % 10 == 0:
                train_acc, train_loss = evaluate(model, train_data, args, "train", plot=False)
                test_acc, test_loss = evaluate(model, test_data, args, "test", plot=False)
                train_accs.append(train_acc)
                train_losses.append(train_loss)
                test_accs.append(test_acc)
                test_losses.append(test_loss)

                if verbose:
                    print(("... eval train_model_swag_binning:", ep, t, train_acc, test_acc))
                    sys.stdout.flush()

            xs = xs.to(device)
            ys = ys.to(device)

            opt.zero_grad()

            preds, max_prob = model(xs)
            l_sup = nn.functional.cross_entropy(preds, ys, reduction="mean")

            # MI upper bound
            zs_dropout = model(xs, repr=True)
            l_MIs_batch = []
            for z_i, z in enumerate(zs_dropout): # bn, z_sz
                assert len(z.shape) == 2
                if z_i in args.MI_reg_layers:
                    noise = torch.randn(z.shape, device=device) * np.sqrt(1./ (2. * np.pi * np.exp(1))) # stddev
                    z_noise = z + noise
                    l_MI_i = torch.linalg.vector_norm(z_noise, ord=2, dim=1)
                    assert l_MI_i.shape == (xs.shape[0],)
                    l_MIs_batch.append(l_MI_i)

            l_MI = torch.cat(l_MIs_batch).mean()

            maxes.append(max_prob)

            MI, ranges, bin_szs = est_MI_binning(model, MI_data.dataset, num_bins=args.num_bins)
            print(("Time %s: bin MI %s, range %s, bin_szs %s, loss: main %s, MI %s (%s)" %
                   (t, MI, ranges, bin_szs, l_sup.item(), (lamb_lr * l_MI.item()), l_MI.item())))
            sys.stdout.flush()

            binning_MIs.append(MI)
            l_MIs.append(l_MI.item())

            (l_sup + lamb_lr * l_MI).backward()

            opt.step()

            if ep == 0: analyse(model, grads=True, t=t)

            t += 1

    # Final eval
    train_acc, train_loss = evaluate(model, train_data, args, "train", plot=False)
    test_acc, test_loss = evaluate(model, test_data, args, "test", plot=False)
    train_accs.append(train_acc)
    train_losses.append(train_loss)
    test_accs.append(test_acc)
    test_losses.append(test_loss)

    MI_mc_perlayer, ranges, bin_szs = est_MI_binning(model, MI_data.dataset, num_bins=args.num_bins)
    MI_cond_mc_perlayer = est_MI_binning_cond(model, MI_data.dataset, args.C, num_bins_cond=args.num_bins_cond)

    #MI_cond_mc = est_MI_cond(model, args.C, MI_data, jensen=False).item()

    #MI_jensen = est_MI(model, MI_data.dataset, sz=min(MI_SZ, len(MI_data.dataset)), jensen=True, requires_grad=False).item()
    #MI_cond_jensen = est_MI_cond(model, args.C, MI_data, jensen=True).item()

    swag_model.sample(0.0)
    test_acc_swag, test_loss_swag = evaluate(swag_model, test_data, args, "test", plot=False)

    diagnostics = {"train_losses": train_losses,
                   "train_accs": train_accs,
                   "test_losses": test_losses,
                   "test_accs": test_accs,
                   "test_loss_swag": test_loss_swag,
                   "test_acc_swag": test_acc_swag,

                   "MI_mc_perlayer": MI_mc_perlayer,
                   "MI_mc_perlayer_last": MI_mc_perlayer[-1],

                   "ranges": ranges,
                   "bin_szs": bin_szs,
                   "MI_cond_mc_perlayer": MI_cond_mc_perlayer,
                   "MI_cond_mc_perlayer_last": MI_cond_mc_perlayer[-1],

                   #"MI_cond_mc": MI_cond_mc,
                   #"MI_jensen": MI_jensen,
                   #"MI_cond_jensen": MI_cond_jensen,

                   "binning_MIs": binning_MIs,
                   "l_MIs": l_MIs,
                   "maxes": maxes,
                   }

    return model, swag_model, diagnostics



"""

def compute_MI_theta_D_single_seed_jensen(swag_models, num_samples):

    # uses covariance
    num_data_instances = len(swag_models)
    log_ratios = [[], []]

    for d_i in range(num_data_instances):
        swag = swag_models[d_i]
        for i in range(num_samples):
            # sample theta from model, compute log prob (P(theta | D))
            swag.sample(scale=1.0, cov=True, block=True)  # populates param value fields

            for last_layer in [0, 1]:
                try:
                    #log_posterior = swag.compute_logprob_cumu(last_layer, block=True)  # scalar
                    log_posterior = swag.compute_logprob_cumu(last_layer, block=True)  # scalar todo

                    param_list = swag.get_cumu_param_list(last_layer)

                    # compute average (non-log) prob over all models, take log
                    logprob = [log_posterior]
                    for other_swag in swag_models[:d_i] + swag_models[(d_i + 1):]:
                        other_log_prob = other_swag.compute_logprob_cumu(last_layer, vec=param_list, block=True)
                        logprob.append(other_log_prob)

                    logprob = torch.stack(logprob)

                    log_prior = logprob.mean()

                    assert len(log_prior.shape) == 0 and len(log_posterior.shape) == 0
                    log_ratios[last_layer].append(log_posterior - log_prior)
                except:
                    print(traceback.format_exc())
                    print("Skipping sample for %s" % str((d_i, i)))

    #per_layer_MI = [torch.stack(log_ratios[i]) for i in [0, 1]]
    #per_layer_MI = torch.stack(per_layer_MI, dim=0)
    #assert per_layer_MI.shape == (2, num_samples * num_data_instances,)
    #return per_layer_MI.mean(dim=1).cpu()

    per_layer_MIs = torch.zeros(2, dtype=torch.float)
    for last_layer in [0, 1]:
        if len(log_ratios[last_layer]) > 0:
            per_layer_MI_curr = torch.stack(log_ratios[last_layer])
            if not per_layer_MI_curr.shape == (num_samples * num_data_instances,): print("Warning, expected %s, got %s" % ((num_samples * num_data_instances,), per_layer_MI_curr.shape))
            print(per_layer_MI_curr)
            per_layer_MIs[last_layer] = per_layer_MI_curr.mean().item()
        else:
            per_layer_MIs[last_layer] = np.nan
    return per_layer_MIs

def compute_MI_theta_D_multiseed_jensen(swag_models, num_samples):

    # uses covariance
    num_data_instances = len(swag_models[0])
    num_seeds = len(swag_models)
    log_ratios = [[], []]

    for d_i in range(num_data_instances):
        for seed in range(num_seeds):
            swag = swag_models[seed][d_i]
            for i in range(num_samples):

                try:

                    # sample theta from model, compute log prob (P(theta | D))
                    swag.sample(scale=1.0, cov=True, block=True)  # populates param value fields
                    for last_layer in [0, 1]:
                        param_list = swag.get_cumu_param_list(last_layer)

                        logprob = torch.zeros(num_data_instances, num_seeds, device=device)

                        for d_j in range(num_data_instances):
                            for other_seed in range(num_seeds):
                                other_swag = swag_models[other_seed][d_j]
                                other_log_prob = other_swag.compute_logprob_cumu(last_layer, vec=param_list, block=True)
                                logprob[d_j, other_seed] = other_log_prob

                        log_posterior = logprob[d_i].mean()  # est - over all seeds
                        log_prior = logprob.mean()  # est - over all data instances and seeds

                        assert len(log_prior.shape) == 0 and len(log_posterior.shape) == 0
                        log_ratios[last_layer].append(log_posterior - log_prior)

                except:
                    print(traceback.format_exc())
                    print("Skipping sample for %s" % str((d_i, i)))

    #per_layer_MI = [torch.stack(log_ratios[i]) for i in [0, 1]]
    #per_layer_MI = torch.stack(per_layer_MI, dim=0)
    #assert per_layer_MI.shape == (2, num_samples * num_data_instances * num_seeds,)
    #return per_layer_MI.mean(dim=1).cpu()

    per_layer_MIs = torch.zeros(2, dtype=torch.float)
    for last_layer in [0, 1]:
        if len(log_ratios[last_layer]) > 0:
            per_layer_MI_curr = torch.stack(log_ratios[last_layer])
            if not per_layer_MI_curr.shape == (num_samples * num_data_instances,): print("Warning, expected %s, got %s" % ((num_samples * num_data_instances,), per_layer_MI_curr.shape))
            print(per_layer_MI_curr)
            per_layer_MIs[last_layer] = per_layer_MI_curr.mean().item()
        else:
            per_layer_MIs[last_layer] = np.nan
    return per_layer_MIs

"""


def compute_MI_theta_D_single_seed_jensen(swag_models, num_samples, layers):
    # uses covariance
    num_data_instances = len(swag_models)
    log_ratios = [[] for _ in layers]

    for d_i in range(num_data_instances):
        swag = swag_models[d_i]
        for i in range(num_samples):
            # sample theta from model, compute log prob (P(theta | D))
            swag.sample(scale=1.0, cov=True, block=True)  # populates param value fields

            for last_layer in layers:
                log_posterior = swag.compute_logprob_cumu(last_layer, block=True)

                param_list = swag.get_cumu_param_list(last_layer)

                # compute average (non-log) prob over all models, take log
                logprob = [log_posterior]
                for other_swag in swag_models[:d_i] + swag_models[(d_i + 1):]:
                    other_log_prob = other_swag.compute_logprob_cumu(last_layer, vec=param_list, block=True)
                    logprob.append(other_log_prob)

                logprob = torch.stack(logprob)

                log_prior = logprob.mean()

                assert len(log_prior.shape) == 0 and len(log_posterior.shape) == 0
                log_ratios[last_layer].append(log_posterior - log_prior)

    per_layer_MI = [torch.stack(log_ratios[i]) for i in layers]
    per_layer_MI = torch.stack(per_layer_MI, dim=0)
    assert per_layer_MI.shape == (len(layers), num_samples * num_data_instances,)
    return per_layer_MI.mean(dim=1).cpu()


def compute_MI_theta_D_multiseed_jensen(swag_models, num_samples, layers):

    # uses covariance
    num_data_instances = len(swag_models[0])
    num_seeds = len(swag_models)
    log_ratios = [[] for _ in layers]

    for d_i in range(num_data_instances):
        for seed in range(num_seeds):
            swag = swag_models[seed][d_i]
            for i in range(num_samples):

                # sample theta from model, compute log prob (P(theta | D))
                swag.sample(scale=1.0, cov=True, block=True)  # populates param value fields
                for last_layer in layers:
                    param_list = swag.get_cumu_param_list(last_layer)

                    logprob = torch.zeros(num_data_instances, num_seeds, device=device)

                    for d_j in range(num_data_instances):
                        for other_seed in range(num_seeds):
                            other_swag = swag_models[other_seed][d_j]
                            other_log_prob = other_swag.compute_logprob_cumu(last_layer, vec=param_list, block=True)
                            logprob[d_j, other_seed] = other_log_prob

                    log_posterior = logprob[d_i].mean()  # est - over all seeds
                    log_prior = logprob.mean()  # est - over all data instances and seeds

                    assert len(log_prior.shape) == 0 and len(log_posterior.shape) == 0
                    log_ratios[last_layer].append(log_posterior - log_prior)

    per_layer_MI = [torch.stack(log_ratios[i]) for i in layers]
    per_layer_MI = torch.stack(per_layer_MI, dim=0)
    assert per_layer_MI.shape == (len(layers), num_samples * num_data_instances * num_seeds,)
    return per_layer_MI.mean(dim=1).cpu()


def get_key(arch_i, decay_i, data_i, seed_i, to_vary):
    if to_vary == "none":
        return 0

    return {"arch": arch_i, "decay": decay_i, "data_instance": data_i}[to_vary]


def get_key_MNIST(arch_i, decay_i, batch_i, data_i, seed_i, to_vary):
    if to_vary == "none":
        return 0

    return {"arch": arch_i, "decay": decay_i, "batch_i": batch_i, "data_instance": data_i}[to_vary]



def whiten(x):
    assert isinstance(x, np.ndarray) and len(x.shape) == 1
    res = (x - x.mean()) / x.std()
    assert x.shape == res.shape
    return res


def padname(s):
    return s + " " * (36 - len(s))


def arch_name(arch):
    assert isinstance(arch, list)
    return str(max(arch))

    # todo
    arch_s = str(arch)
    d = {
        "[2, 256, 256, 128, 128, 5]": "256",
        "[2, 128, 128, 64, 64, 5]": "128",
        "[2, 64, 64, 32, 32, 5]": "64",
        "[2, 32, 32, 16, 16, 5]": "32",
    }
    return d[arch_s]


def arch_name_MNIST(arch):
    assert isinstance(arch, list)
    return str(max(arch))

    """
    arch_s = str(arch)
    d = {
    "[1, 128, 1028, 1028, 1028, 10]": 1028,
    "[1, 64, 512, 512, 512, 10]": 512,
    "[1, 32, 256, 256, 256, 10]": 256,
    "[1, 16, 128, 128, 128, 10]": 128,
    }
    return d[arch_s]
    """
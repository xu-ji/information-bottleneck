import random
import numpy as np
import torch
import gc
from collections import defaultdict
import torch.nn.functional as F
import os
from datetime import datetime
import sys
import pandas as pd

device = torch.device("cuda:0")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def exp_name_flat(dataset, seed, batch_size, wd, model):
    setting_name = "%s_%s_%s" % (batch_size, wd, model)
    return "%s__seed_%s__sett_%s" % (dataset, seed, setting_name)


def exp_name(args): # used for both swag and our scripts
    setting_name = "%s_%s_%s" % (args.batch_size, args.wd, args.model)
    return "%s__seed_%s__sett_%s" % (args.dataset, args.seed, setting_name)


def dataset_name(data, seed, i):
    return "%s_seed_%d_inst_%d" % (data, seed, i)


def assess(model, train_dl):
    model.eval()
    accs = []
    losses = []
    with torch.no_grad():
        for xs, ys in train_dl:
            xs, ys = xs.to(device), ys.to(device)
            preds = model(xs)
            accs.append((preds.argmax(dim=1) == ys).to(torch.float))
            losses.append(F.cross_entropy(preds, ys, reduction="none"))
    return 1 - torch.cat(accs, dim=0).mean().item(), torch.cat(losses, dim=0).mean().item()


def clean(s):
    return s.replace("_", " ")


def setting_diff(tup1, tup2):
    assert len(tup1) == len(tup2)
    diff = 0
    for i in range(len(tup1)):
        if tup1[i] != tup2[i]:
            diff += 1
    return diff


def diff_axis(tup1, tup2):
    for i in range(len(tup1)):
        if tup1[i] != tup2[i]:
            return i


def compute_MI_theta_D_single_seed_ls(dataset_root, seed, batch_size, wd, model_name,
                       res_dir, data_instances, num_samples, num_layers):

    swags = []
    for d_i in data_instances:
        dataset = "%s" % dataset_name(dataset_root, seed, d_i)
        path = os.path.join(res_dir, dataset,
                            "%s.pt" % exp_name_flat(dataset, seed, batch_size, wd, model_name))
        print(path)
        if not os.path.exists(path):
            assert False
        saved = torch.load(path)
        swag_model = saved["swag_model"]
        swags.append(swag_model)

    log_ratios = torch.zeros(num_layers, num_samples * len(data_instances), device=device)
    j = 0
    for d_i in range(len(data_instances)):
        swag = swags[d_i]
        for i in range(num_samples):
            print(("in loop", d_i, seed, i, datetime.now()))
            sys.stdout.flush()
            # sample theta from model, compute log prob (P(theta | D))
            swag.sample(scale=1.0, cov=True, block=True) # populates param value fields

            for last_layer in range(num_layers):
                log_posterior = swag.compute_logprob_cumu(last_layer, block=True) # scalar
                param_list = swag.get_cumu_param_list(last_layer)

                # compute average (non-log) prob over all models, take log
                logprob = [log_posterior]
                for other_swag in swags[:d_i] + swags[(d_i+1):]:
                    other_log_prob = other_swag.compute_logprob_cumu(last_layer, vec=param_list, block=True)
                    logprob.append(other_log_prob)

                logprob = torch.stack(logprob)
                log_prior = - np.log(len(data_instances)) + torch.logsumexp(logprob, dim=0)

                assert len(log_prior.shape) == 0 and len(log_posterior.shape) == 0
                log_ratios[last_layer][j] = log_posterior - log_prior

            j += 1

    per_layer_MI = log_ratios.mean(dim=1)
    return per_layer_MI.cpu()


def compute_MI_theta_D_single_seed_kol(dataset_root, seed, batch_size, wd, model_name,
                       res_dir, data_instances, num_samples, num_layers):

    means = [[] for _ in range(num_layers)]
    varis = [[] for _ in range(num_layers)]
    for d_i in data_instances:
        dataset = "%s" % dataset_name(dataset_root, seed, d_i)
        path = os.path.join(res_dir, dataset,
                            "%s.pt" % exp_name_flat(dataset, seed, batch_size, wd, model_name))
        print(path)
        if not os.path.exists(path):
            assert False
        saved = torch.load(path)
        swag_model = saved["swag_model"]  # swag object with posterior

        for l in range(num_layers):
            mean, cov = swag_model.mean_diag_cov_cumu(l) # populates param fields with mean
            assert mean.shape == cov.shape
            if d_i == 0: print((mean.shape, cov.shape))
            means[l].append(mean)
            varis[l].append(cov)

    MIs = torch.zeros(num_layers)
    for l in range(num_layers):
        kls = torch.zeros(len(data_instances), len(data_instances))

        for d_i in range(len(data_instances)):
            for d_j in range(len(data_instances)):
                # KL (p_i || p_j)
                mu_i = means[l][d_i]
                mu_j = means[l][d_j]
                v_i = varis[l][d_i]
                v_j = varis[l][d_j]

                inv_v_j = 1. / v_j

                kl_i_j = 0.5 * (v_j.log().sum() - v_i.log().sum() + \
                    ((mu_i - mu_j) * (mu_i - mu_j) * inv_v_j).sum()  + \
                    (inv_v_j * v_i).sum() - mu_i.shape[0])

                kls[d_i, d_j] = kl_i_j.item()

        print(("kl", l, kls))
        MIs[l] = - (- np.log(len(data_instances)) + torch.logsumexp(- kls, dim=1)).mean(dim=0)

    return MIs.cpu()


def compute_MI_theta_D_single_seed_jensen(dataset_root, seed, batch_size, wd, model_name,
                       res_dir, data_instances, num_samples, num_layers):

    swags = []
    for d_i in data_instances:
        dataset = "%s" % dataset_name(dataset_root, seed, d_i)
        path = os.path.join(res_dir, dataset,
                            "%s.pt" % exp_name_flat(dataset, seed, batch_size, wd, model_name))
        print(path)
        if not os.path.exists(path):
            assert False
        saved = torch.load(path)
        swag_model = saved["swag_model"]  # swag object with posterior
        swags.append(swag_model)

    log_ratios = torch.zeros(num_layers, num_samples * len(data_instances), device=device)
    j = 0
    for d_i in range(len(data_instances)):
        swag = swags[d_i]
        for i in range(num_samples):
            print(("in loop", d_i, seed, i, datetime.now()))
            sys.stdout.flush()
            # sample theta from model, compute log prob (P(theta | D))
            swag.sample(scale=1.0, cov=True, block=True) # populates param value fields

            for last_layer in range(num_layers):
                log_posterior = swag.compute_logprob_cumu(last_layer, block=True) # scalar
                param_list = swag.get_cumu_param_list(last_layer)

                # compute average (non-log) prob over all models, take log
                logprob = [log_posterior]
                for other_swag in swags[:d_i] + swags[(d_i+1):]:
                    other_log_prob = other_swag.compute_logprob_cumu(last_layer, vec=param_list, block=True)
                    logprob.append(other_log_prob)

                log_prior = torch.stack(logprob).mean() # average in log domain

                assert len(log_prior.shape) == 0 and len(log_posterior.shape) == 0
                log_ratios[last_layer][j] = log_posterior - log_prior

            j += 1

    per_layer_MI = log_ratios.mean(dim=1)
    return per_layer_MI.cpu()


def compute_MI_theta_D_multiseed(dataset_root, num_seeds, batch_size, wd, model_name,
                       res_dir, data_instances, num_samples, num_layers):
    swags = [[] for _ in range(len(data_instances))] # data instance, seed
    for d_i in data_instances:
        for seed in range(num_seeds):
            dataset = "%s" % dataset_name(dataset_root, seed, d_i)
            path = os.path.join(res_dir, dataset,
                                "%s.pt" % exp_name_flat(dataset, seed, batch_size, wd, model_name))
            print(path)
            if not os.path.exists(path):
                assert False
            saved = torch.load(path)
            swag_model = saved["swag_model"]  # swag object with posterior
            swags[d_i].append(swag_model)

    log_ratios = torch.zeros(num_layers, num_samples * len(data_instances) * num_seeds, device=device)
    j = 0
    all_probs = []
    for d_i in range(len(data_instances)):
        for seed in range(num_seeds):
            swag = swags[d_i][seed]
            for i in range(num_samples):
                print(("in loop", d_i, seed, i, datetime.now()))
                sys.stdout.flush()
                # sample theta from model, compute log prob (P(theta | D))
                swag.sample(scale=1.0, cov=True, block=True) # populates param value fields

                for last_layer in range(num_layers):
                    #log_prob = swag.compute_logprob_cumu(last_layer, block=True) # scalar
                    param_list = swag.get_cumu_param_list(last_layer) # indexed from 0 as first
                    logprob = torch.zeros(len(data_instances), num_seeds, device=device)

                    for d_j in range(len(data_instances)):
                        for other_seed in range(num_seeds):
                            other_swag = swags[d_j][other_seed]
                            other_log_prob = other_swag.compute_logprob_cumu(last_layer, vec=param_list, block=True)
                            logprob[d_j, other_seed] = other_log_prob

                    logprob_posterior = logprob[d_i].mean() # est - over all seeds
                    logprob_prior = logprob.mean() # est - over all data instances and seeds

                    assert len(logprob_posterior.shape) == 0 and len(logprob_prior.shape) == 0

                    # subtract logs
                    log_ratios[last_layer, j] = logprob_posterior - logprob_prior
                j += 1

    assert j == num_samples * len(data_instances) * num_seeds

    per_layer_MI = log_ratios.mean(dim=1)

    return per_layer_MI.cpu(), all_probs


def compute_MI_Z_Xy(model, train_dl, num_layers, num_classes, stds, sz1, sz2, batch_sz, mode):
    num_layers_incl_in = num_layers + 1
    # use variance bound
    feats = [[[] for _ in range(num_layers_incl_in)] for _ in range(num_classes)]
    y_counts = torch.zeros(num_classes)
    with torch.no_grad():
        for j, (xs, ys) in enumerate(train_dl): # was train_dl_all
            xs = xs.to(device)
            ys = ys.to(device)
            _, all_out = model(xs, return_features=True)  # num layers: batch sz, repr sz
            for y in range(num_classes):
                matches = ys == y
                for l in range(num_layers_incl_in):
                    feats[y][l].append(
                        all_out[l][matches, :])  # y, num_layers: list of batch_sz, repr sz
                y_counts[y] += matches.sum().item()

        feat_ent_bounds_i = torch.zeros(num_layers_incl_in, num_classes)
        c_l_y = torch.zeros(num_layers_incl_in, num_classes)
        for y in range(num_classes):
            for l in range(num_layers_incl_in):
                log_post = torch.distributions.normal.Normal(0, stds[l].item()).log_prob(torch.tensor([0])).item()

                feats_y_l = torch.cat(feats[y][l], dim=0)  # num samples, repr sz
                assert len(feats_y_l.shape) == 2

                # MI
                num_samples, repr_sz = feats_y_l.shape
                eval_i = torch.tensor(np.random.choice(num_samples, sz1, replace=True)) # device=device
                eval_feats = feats_y_l[eval_i] # sz, repr_sz

                ref_i =  torch.tensor(np.random.choice(num_samples, sz2, replace=True)) # device=device
                ref_feats = feats_y_l[ref_i]

                distr = torch.distributions.normal.Normal(ref_feats, stds[l].item())
                log_prior = compute_log_prior(sz1, batch_sz, eval_feats, repr_sz, distr, sz2, mode)
                feat_ent_bounds_i[l, y] = (log_post - log_prior).mean()

                # c_l_y
                # for each sample, pick another sample and an index, make the swap
                swapped_feats = eval_feats.clone() # sz, repr_sz

                swap_i = torch.tensor(np.random.choice(sz1, sz1, replace=True)) # device=device
                swap_j = torch.tensor(np.random.choice(repr_sz, sz1, replace=True))

                swapped_feats[range(sz1), swap_j] = eval_feats[swap_i, swap_j]

                new_log_prior = compute_log_prior(sz1, batch_sz, swapped_feats, repr_sz, distr, sz2, mode)
                max_diff = (log_prior - new_log_prior).abs().max()
                c_l_y[l, y] = max_diff.item()

    return feat_ent_bounds_i, c_l_y, y_counts


def compute_MI_Z_X(model, train_dl, num_layers, stds, sz1, sz2, batch_sz, mode):

    num_layers_incl_in = num_layers + 1

    assert stds.shape == (num_layers_incl_in,)

    feats = [[] for _ in range(num_layers_incl_in)]
    with torch.no_grad():
        for j, (xs, ys) in enumerate(train_dl): # was train_dl_all
            xs = xs.to(device)
            _, all_out = model(xs, return_features=True)  # num layers: batch sz, repr sz
            for l in range(num_layers_incl_in):
                feats[l].append(all_out[l])  # num_layers: list of batch_sz, repr sz

        log_post = torch.distributions.normal.Normal(0, stds).log_prob(torch.zeros(num_layers_incl_in))
        assert log_post.shape == (num_layers_incl_in,)

        feat_ent_bounds_i = torch.zeros(num_layers_incl_in)
        layer_logprob = torch.zeros(num_layers_incl_in)

        for l in range(num_layers_incl_in):
            feats_l = torch.cat(feats[l], dim=0)  # num samples, repr sz (float)
            assert len(feats_l.shape) == 2
            num_samples, repr_sz = feats_l.shape

            eval_i = torch.tensor(np.random.choice(num_samples, sz1, replace=True))  # device=device
            eval_feats = feats_l[eval_i]  # sz, repr_sz

            ref_i = torch.tensor(np.random.choice(num_samples, sz2, replace=True))  # device=device
            ref_feats = feats_l[ref_i]

            distr = torch.distributions.normal.Normal(ref_feats, stds[l].item())
            log_prior = compute_log_prior(sz1, batch_sz, eval_feats, repr_sz, distr, sz2, mode)

            layer_logprob[l] = log_prior.mean().item()
            feat_ent_bounds_i[l] = (log_post[l] - log_prior).mean()

    return feat_ent_bounds_i, layer_logprob


def compute_log_prior(sz1, batch_sz, eval_feats, repr_sz, distr, sz2, mode):
    nb = int(np.ceil(sz1 / float(batch_sz)))
    log_priors = []
    for b_i in range(nb):
        s = (b_i * batch_sz)
        e =  min((b_i + 1) * batch_sz, sz1)
        b_sz = e - s

        eval_feats_b = eval_feats[s:e]
        eval_feats_exp = eval_feats_b.unsqueeze(1).expand(b_sz, sz2, repr_sz)  # duplicate along 2nd dim
        log_prob = distr.log_prob(eval_feats_exp)
        assert log_prob.shape == (b_sz, sz2, repr_sz)

        log_prob = log_prob.sum(dim=2)  # log P(z_l | x) for each z_l (dim=0) and x or ref z_l (dim=1)

        if mode == "jensen":
            log_prior = log_prob.mean(dim=1)
        elif mode == "mc":
            log_prior =  - np.log(sz2) + torch.logsumexp(log_prob, dim=1)
        else:
            raise NotImplementedError

        log_priors.append(log_prior)

    return torch.cat(log_priors, dim=0) # sz1


def compute_max_acts(model, train_dl, num_layers):
    num_layers_incl_in = num_layers + 1
    max_acts = torch.zeros(num_layers_incl_in)
    std_acts = torch.zeros(num_layers_incl_in)
    mean_acts = torch.zeros(num_layers_incl_in)

    with torch.no_grad():
        for j, (xs, ys) in enumerate(train_dl):  # was train_dl_all
            xs = xs.to(device)
            # ys = ys.to(device)
            _, all_out = model(xs, return_features=True)  # num layers: batch sz, repr sz
            for l in range(num_layers_incl_in):
                max_acts[l] = max(all_out[l].max().item(), max_acts[l])
                std_acts[l] += all_out[l].std().item()
                mean_acts[l] += all_out[l].mean().item()

    return max_acts, std_acts / len(train_dl), mean_acts / len(train_dl)


def compute_factors(model, train_dl, num_layers, num_classes, stds, sz1, sz2, batch_sz, mode):
    num_layers_incl_in = num_layers + 1
    # use variance bound
    feats = [[[] for _ in range(num_layers_incl_in)] for _ in range(num_classes)]
    y_counts = torch.zeros(num_classes)
    with torch.no_grad():
        for j, (xs, ys) in enumerate(train_dl): # was train_dl_all
            xs = xs.to(device)
            ys = ys.to(device)
            _, all_out = model(xs, return_features=True)  # num layers: batch sz, repr sz
            for y in range(num_classes):
                matches = ys == y
                for l in range(num_layers_incl_in):
                    feats[y][l].append(
                        all_out[l][matches, :])  # y, num_layers: list of batch_sz, repr sz
                y_counts[y] += matches.sum().item()

        feat_ent_bounds_i = torch.zeros(num_layers_incl_in, num_classes)
        c_l_y = torch.zeros(num_layers_incl_in, num_classes)
        for y in range(num_classes):
            for l in range(num_layers_incl_in):
                log_post = torch.distributions.normal.Normal(0, stds[l].item()).log_prob(torch.tensor([0])).item()

                feats_y_l = torch.cat(feats[y][l], dim=0)  # num samples, repr sz
                assert len(feats_y_l.shape) == 2

                # MI
                num_samples, repr_sz = feats_y_l.shape
                eval_i = torch.tensor(np.random.choice(num_samples, sz1, replace=True)) # device=device
                eval_feats = feats_y_l[eval_i] # sz, repr_sz

                ref_i =  torch.tensor(np.random.choice(num_samples, sz2, replace=True)) # device=device
                ref_feats = feats_y_l[ref_i]

                distr = torch.distributions.normal.Normal(ref_feats, stds[l].item())
                log_prior = compute_log_prior(sz1, batch_sz, eval_feats, repr_sz, distr, sz2, mode)
                feat_ent_bounds_i[l, y] = (log_post - log_prior).mean()

                # c_l_y
                # for each sample, pick another sample and an index, make the swap
                swapped_feats = eval_feats.clone() # sz, repr_sz

                swap_i = torch.tensor(np.random.choice(sz1, sz1, replace=True)) # device=device
                swap_j = torch.tensor(np.random.choice(repr_sz, sz1, replace=True))

                swapped_feats[range(sz1), swap_j] = eval_feats[swap_i, swap_j]

                new_log_prior = compute_log_prior(sz1, batch_sz, swapped_feats, repr_sz, distr, sz2, mode)
                max_diff = (log_prior - new_log_prior).abs().max()
                c_l_y[l, y] = max_diff.item()

    return feat_ent_bounds_i, c_l_y, y_counts



def normalize(v): # [0, 1]
    if v.isfinite().sum() == 0:
        return v

    m = v[v.isfinite()].min()
    v = v - m

    m = v[v.isfinite()].max()

    v = v / m
    return v


def summarize_metrics(train_err, train_loss, test_err, test_loss, gen_err, gen_loss, per_layer_metrics, per_layer_names, num_layers):
    metrics = [
        train_err,
        train_loss,

        test_err,
        test_loss,

        gen_err,
        gen_loss
    ]

    for m in per_layer_metrics:
        assert m.shape == (num_layers,)
        metrics.append(m.mean())
        metrics.append(m.min())
        metrics.append(m.max())
        metrics.append(m[0])
        metrics.append(m[-1])

    return torch.tensor(metrics)


def to_bits(x):
    return x * np.log(2)


def to_cpu(x):
    if isinstance(x, list):
        res = []
        for f in x:
            assert isinstance(f, torch.Tensor)
            res.append(f.cpu())
    elif isinstance(x, torch.Tensor):
        res = x.cpu()
    else:
        print(x.__class__)
        assert False

    return res


def create_dataframe(GL_curr, v, archs):
    table = []
    assert len(GL_curr.shape) == 1 and GL_curr.shape == v.shape
    assert len(archs) == GL_curr.shape[0]
    for i in range(GL_curr.shape[0]):
        table.append([GL_curr[i], v[i], archs[i]])

    return pd.DataFrame(table, columns=["GL_curr", "v", "arch"])

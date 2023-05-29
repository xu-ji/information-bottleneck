import torch
import os
from .util import *
from swag.data import load_instance
import scipy.stats as stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import itertools
from datetime import datetime
import argparse
import seaborn as sns

args_form = argparse.ArgumentParser(allow_abbrev=False)

args_form.add_argument("--out_dir", type=str, default="/network/scratch/x/xu.ji/mutual_info/v7/")

args_form.add_argument("--compute_MI_theta_D_only", default=False, action="store_true")
args_form.add_argument("--compute_MI_theta_D_only_start", type=int, default=0)
args_form.add_argument("--compute_MI_theta_D_only_end", type=int, default=0)
args_form.add_argument("--posterior_mode", type=str, choices=["jensen", "jensen_multiseed"], default="jensen_multiseed")

args_form.add_argument("--compute_MI_X_Z_only", default=False, action="store_true")
args_form.add_argument("--compute_MI_X_Z_only_start", type=int, default=0)
args_form.add_argument("--compute_MI_X_Z_only_end", type=int, default=0)
args_form.add_argument("--MI_X_Z_mode", type=str, default="mle")
args_form.add_argument("--MI_X_Z_linear_base", type=float, default=0)

args_form.add_argument("--compute_performance", default=False, action="store_true")
args_form.add_argument("--compute_performance_start", type=int, default=0)
args_form.add_argument("--compute_performance_end", type=int, default=0)

args_form.add_argument("--compute_metrics", default=False, action="store_true")
args_form.add_argument("--compute_inds_start", type=int, default=0)
args_form.add_argument("--compute_inds_end", type=int, default=0)

args = args_form.parse_args()

set_seed(0)

dataset_root = "CIFAR10"
num_classes = 10

data_dir = os.path.join("/network/scratch/x/xu.ji/datasets/", dataset_root)

num_samples = 5 # per swag model in the posterior
data_instances = list(range(5))
num_layers = 5
num_seeds = 4

sz1 = 2000
sz1_y = 1000
sz2 = 400
batch_sz = 70

batch_sizes = [64, 128, 1024]
decays = [1e-5, 0.0001, 0.001]
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

num_settings = len(settings)
print("num settings: %s" % num_settings)

num_layers_incl_fst = num_layers + 1

stds = torch.tensor([0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 3.0,
                     6.0, 9.0, 12.0, 15.0, 18.0, 21.0])
assert (stds.argsort() == torch.arange(len(stds))).all().item()

colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]

lines = ["dotted", "dashed", "solid"]

setting_names = ["Batch Size", "Weight Decay", "Model"]

marker = itertools.cycle(('o', '*', '+', '.'))  # ',',


################################
# Model compression
################################


if args.compute_MI_theta_D_only:
    print("(compute_MI_theta_D_only)")
    MI_theta_D_out_dir = os.path.join(args.out_dir, "MI_theta_D_%s" % args.posterior_mode)
    os.makedirs(MI_theta_D_out_dir)

    if args.posterior_mode == "jensen_multiseed":
        curr_i = 0
        for s_i, (batch_size, wd, model_name) in enumerate(settings):
            if curr_i in list(
                range(args.compute_MI_theta_D_only_start, args.compute_MI_theta_D_only_end)):
                print("Doing setting %s %s, %s" % (s_i, str(settings[s_i]), datetime.now()))
                sys.stdout.flush()
                theta_D_key = "%s_%s_%s_%s" % (dataset_root, batch_size, wd, model_name)
                theta_D_p = os.path.join(MI_theta_D_out_dir, "theta_D_key_%s.pt" % theta_D_key)
                print("Computing theta_D for %s from scratch" % theta_D_key)
                MI_theta_D, all_probs = compute_MI_theta_D_multiseed(dataset_root, num_seeds,
                                                                     batch_size, wd, model_name,
                                                                     args.out_dir, data_instances,
                                                                     num_samples,
                                                                     num_layers)  # L
                MI_theta_D = torch.cat((torch.zeros(1), MI_theta_D),
                                       dim=0)  # first entry is empty network
                assert MI_theta_D.shape == (num_layers_incl_fst,)
                torch.save((MI_theta_D, all_probs), theta_D_p)

            curr_i += 1

    else:
        curr_i = 0
        for seed in range(num_seeds):
            for s_i, (batch_size, wd, model_name) in enumerate(settings):
                if curr_i in list(range(args.compute_MI_theta_D_only_start, args.compute_MI_theta_D_only_end)):
                    print("Doing setting %s %s, %s" % (s_i, str(settings[s_i]), datetime.now()))
                    sys.stdout.flush()
                    # MI(theta ; D) [shared across instances of D] per l
                    # shared among seeds AND data instances
                    theta_D_key = "%s_%s_%s_%s_%s" % (dataset_root, seed, batch_size, wd, model_name)
                    theta_D_p = os.path.join(MI_theta_D_out_dir, "theta_D_key_%s.pt" % theta_D_key)
                    print("Computing theta_D for %s from scratch" % theta_D_key)
                    if  args.posterior_mode == "kol":
                        MI_fn = compute_MI_theta_D_single_seed_kol
                    elif  args.posterior_mode == "ls":
                        MI_fn = compute_MI_theta_D_single_seed_ls
                    elif args.posterior_mode == "jensen":
                        MI_fn = compute_MI_theta_D_single_seed_jensen
                    else:
                        raise NotImplementedError

                    MI_theta_D = MI_fn(dataset_root, seed, batch_size, wd, model_name,
                                       args.out_dir, data_instances, num_samples, num_layers) # L
                    MI_theta_D = torch.cat((torch.zeros(1), MI_theta_D), dim=0) # first entry is empty network
                    assert MI_theta_D.shape == (num_layers_incl_fst,)
                    torch.save(MI_theta_D, theta_D_p)

                curr_i += 1

        print("final curr_i %d" % curr_i)

    exit(0)


################################
# Representation compression
################################


MI_X_Z_out_dir =  os.path.join(args.out_dir, "MI_X_Z_%s_%s" % (args.MI_X_Z_mode, args.MI_X_Z_linear_base)) # all stored in here
print("MI_X_Z_out_dir %s" % MI_X_Z_out_dir)
if args.compute_MI_X_Z_only:
    print("(compute_MI_X_Z_only)")
    os.makedirs(MI_X_Z_out_dir, exist_ok=True)
    MI_X_Z_compute_inds = list(range(args.compute_MI_X_Z_only_start, args.compute_MI_X_Z_only_end))

    MI_X_Z_compute_ind = 0
    MI_X_Z_violated = 0
    MI_X_Z_skipped = 0
    for seed in range(num_seeds):
        for s_i, (batch_size, wd, model_name) in enumerate(settings):
            for d_i in data_instances:

                if MI_X_Z_compute_ind in MI_X_Z_compute_inds:
                    MI_X_Z_compute_ind_path = os.path.join(MI_X_Z_out_dir, "MI_X_Z_compute_ind_%s.pt" % MI_X_Z_compute_ind)
                    MI_Z_X_final = {}

                    if not os.path.exists(MI_X_Z_compute_ind_path):
                        print("Doing %d: %s, %s, %s (%s)" % (MI_X_Z_compute_ind, seed, str(settings[s_i]), d_i, datetime.now()))
                        sys.stdout.flush()

                        dataset = "%s" % dataset_name(dataset_root, seed, d_i)
                        full_name = exp_name_flat(dataset, seed, batch_size, wd, model_name)
                        path = os.path.join(args.out_dir, dataset, "%s.pt" % full_name)

                        if not os.path.exists(path):
                            print("Skipping... ")
                            sys.stdout.flush()
                            MI_X_Z_skipped += 1

                        else:
                            saved = torch.load(path)
                            model = saved["model"].eval()

                            d, _ = load_instance(dataset, data_dir, batch_size)
                            train_dl, test_dl = d["train"], d["test"]

                            for mode in ["mc", "jensen"]:

                                if args.MI_X_Z_mode == "mle":
                                    MI_Z_X_results = []
                                    logprob_results = []

                                    for std in stds:  # sorted
                                        print((mode, std, datetime.now()))
                                        sys.stdout.flush()
                                        MI_Z_X_std, layer_logprob = compute_MI_Z_X(model, train_dl, num_layers, torch.ones(num_layers_incl_fst).fill_(std), sz1, sz2, batch_sz,
                                                                                   mode=mode)
                                        MI_Z_X_results.append(MI_Z_X_std)
                                        logprob_results.append(layer_logprob)
                                    MI_Z_X_results = torch.stack(MI_Z_X_results, dim=0)  # std_i, L + 1
                                    logprob_results = torch.stack(logprob_results, dim=0)  # std_i, L + 1

                                    # find best satisfying std
                                    best_std_i = torch.zeros(num_layers_incl_fst, dtype=torch.long)

                                    MI_constraint = -np.inf
                                    for curr_l in reversed(range(num_layers_incl_fst)):
                                        std_i_sat_constraint = (MI_Z_X_results[:, curr_l] >= MI_constraint).nonzero(as_tuple=True)[0]  # 1 dim
                                        assert len(std_i_sat_constraint.shape) == 1
                                        if std_i_sat_constraint.shape[0] == 0:
                                            best_std_l_i = logprob_results[:, curr_l].argmax()
                                            MI_X_Z_violated += 1
                                        else:
                                            best_std_l_ii = logprob_results[std_i_sat_constraint, curr_l].argmax()
                                            best_std_l_i = std_i_sat_constraint[best_std_l_ii]

                                        best_std_i[curr_l] = best_std_l_i
                                        MI_constraint = MI_Z_X_results[best_std_l_i, curr_l]

                                    best_stds = stds[best_std_i]

                                    MI_Z_X_final["MI_Z_X_%s" % mode] = MI_Z_X_results[best_std_i, range(num_layers_incl_fst)]
                                    MI_Z_X_final["best_stds_%s" % mode] = best_stds
                                    MI_Z_X_final["best_std_i_%s" % mode] = best_std_i
                                    MI_Z_X_final["MI_Z_X_results_%s" % mode] = MI_Z_X_results
                                    MI_Z_X_final["logprob_results_%s" % mode] = logprob_results

                                    print("mode %s, MI_Z_X %s, avg_MI_Z_Xy %s, \t best_stds %s" % (mode, MI_Z_X_final["MI_Z_X_%s" % mode], MI_Z_X_final["avg_MI_Z_Xy_%s" % mode], MI_Z_X_final["best_stds_%s" % mode]))

                                elif args.MI_X_Z_mode == "linear":
                                    assert args.MI_X_Z_linear_base > 0
                                    max_acts, std_acts, mean_acts = compute_max_acts(model, train_dl, num_layers)
                                    vars = torch.zeros(num_layers_incl_fst).fill_(args.MI_X_Z_linear_base) * max_acts
                                    best_stds = vars.sqrt()

                                    MI_Z_X_linear, layer_logprob = compute_MI_Z_X(model, train_dl, num_layers, best_stds, sz1, sz2, batch_sz, mode=mode)

                                    MI_Z_X_final["MI_Z_X_%s" % mode] = MI_Z_X_linear
                                    MI_Z_X_final["max_acts"] = max_acts
                                    MI_Z_X_final["std_acts"] = std_acts
                                    MI_Z_X_final["mean_acts"] = mean_acts
                                    MI_Z_X_final["logprob_%s" % mode] = layer_logprob

                                MI_Z_Xy, c_l_y, y_counts = compute_MI_Z_Xy(model, train_dl, num_layers, num_classes, stds=best_stds, sz1=sz1_y, sz2=sz2, batch_sz=batch_sz,
                                                                                                mode=mode)  # L + 1, C
                                y_prop = y_counts / y_counts.sum()  # C
                                avg_MI_Z_Xy = (MI_Z_Xy * y_prop.unsqueeze(0)).sum(dim=1)  # L

                                MI_Z_X_final["MI_Z_Xy_%s" % mode] = MI_Z_Xy
                                MI_Z_X_final["c_l_y_%s" % mode] = c_l_y
                                MI_Z_X_final["y_counts_%s" % mode] = y_counts
                                MI_Z_X_final["avg_MI_Z_Xy_%s" % mode] = avg_MI_Z_Xy

                                print("Mode %s, skipped %s, violated %s" % (mode, MI_X_Z_skipped, MI_X_Z_violated))

                            torch.save(MI_Z_X_final, MI_X_Z_compute_ind_path)

                MI_X_Z_compute_ind += 1

    exit(0)


################################
# Performance
################################


performance_out_dir =  os.path.join(args.out_dir, "performance") # all stored in here

if args.compute_performance:
    print("(compute performance)")

    performance_compute_inds = list(range(args.compute_performance_start, args.compute_performance_end))

    performance_compute_ind = 0

    for seed in range(num_seeds):
        for s_i, (batch_size, wd, model_name) in enumerate(settings):
            for d_i in data_instances:

                if performance_compute_ind in performance_compute_inds:
                    performance_path = os.path.join(performance_out_dir, "performance_%s.pt" % performance_compute_ind)
                    performance = {}

                    dataset = "%s" % dataset_name(dataset_root, seed, d_i)
                    full_name = exp_name_flat(dataset, seed, batch_size, wd, model_name)
                    path = os.path.join(args.out_dir, dataset, "%s.pt" % full_name)

                    saved = torch.load(path)
                    model = saved["model"].eval()

                    d, _ = load_instance(dataset, data_dir, batch_size=512)
                    train_dl, test_dl = d["train"], d["test"]
                    num_train = len(train_dl.dataset)

                    train_err, train_loss = assess(model, train_dl)
                    test_err, test_loss = assess(model, test_dl)
                    GE = test_err - train_err
                    GL = test_loss - train_loss

                    print("Doing %d: %s, %s, %s (%s)" % (performance_compute_ind, seed, str(settings[s_i]), d_i, datetime.now()))
                    sys.stdout.flush()

                    performance["train_err"] = train_err
                    performance["train_loss"] = train_loss
                    performance["test_err"] = test_err
                    performance["test_loss"] = test_loss
                    performance["GE"] = GE
                    performance["GL"] = GL
                    performance["num_train"] = num_train

                    torch.save(performance, performance_path)

                performance_compute_ind += 1

    exit(0)


################################
# Results
################################


if args.compute_metrics:
    eps = 1e-6
    vnames_base_invariant = [
        "train_losses",
        "test_losses",
        "train_errs",
        "test_errs",

        "GE",
        "GL",

        "MI_theta_singles_last",
        "MI_theta_multis_last"
    ]

    vnames_base_factors = ["MI_mcs", "MI_cond_mcs", "MI_jensens", "MI_cond_jensens",
        "MI_theta_singles", "MI_theta_multis"]

    vnames_base = vnames_base_invariant + vnames_base_factors

    vnames_combined_noscale = [
        "combined_singles_mc_noscale", "combined_single_conds_mc_noscale",
        "combined_singles_jensen_noscale", "combined_single_conds_jensen_noscale",
        "combined_multis_mc_noscale", "combined_multi_conds_mc_noscale",
        "combined_multis_jensen_noscale", "combined_multi_conds_jensen_noscale",
    ]

    vnames_combined = vnames_combined_noscale

    vnames = []
    for vname in vnames_base:
        vnames.append(vname)
    for vname_combi in vnames_combined:
        vnames.append(vname_combi)

    vnames_perlayer_plot = ["MI_jensens", "MI_cond_jensens", "MI_theta_multis", "combined_multi_conds_jensen_noscale"]

    vnames_pretty = {
    "MI_mcs": r"$\hat{I}(X; Z_l^S)$",
    "MI_cond_mcs": r"$\hat{I}(X; Z_l^S | Y)$",

    "MI_jensens": r"$\breve{I}(X; Z_l^S)$",
    "MI_cond_jensens": r"$\breve{I}(X; Z_l^S | Y)$",

    "MI_theta_singles": r"$\breve{I}({S'}; \theta_l^{S'})$", # jensen upper bound
    "MI_theta_multis": r"$\bar{I}({S'}; \theta_l^{S'})$", # double jensen bound

    "MI_theta_singles_last": r"$\breve{I}(S; \theta_{D+1}^{S'})$", # jensen upper bound
    "MI_theta_multis_last": r"$\bar{I}(S; \theta_{D+1}^{S'})$", # double jensen bound

    "combined_singles_mc_noscale": r"$\breve{I}({S'}; \theta_l^{S'}) + \hat{I}(X; Z_l^S)$",
    "combined_single_conds_mc_noscale": r"$\breve{I}({S'}; \theta_l^{S'}) + \hat{I}(X; Z_l^S | Y)$",

    "combined_singles_jensen_noscale": r"$\breve{I}({S'}; \theta_l^{S'}) + \breve{I}(X; Z_l^S)$",
    "combined_single_conds_jensen_noscale": r"$\breve{I}({S'}; \theta_l^{S'}) + \breve{I}(X; Z_l^S | Y)$",


    "combined_multis_mc_noscale": r"$\bar{I}({S'}; \theta_l^{S'}) + \hat{I}(X; Z_l^S)$",
    "combined_multi_conds_mc_noscale": r"$\bar{I}({S'}; \theta_l^{S'}) + \hat{I}(X; Z_l^S | Y)$",

    "combined_multis_jensen_noscale": r"$\bar{I}({S'}; \theta_l^{S'}) + \breve{I}(X; Z_l^S)$",
    "combined_multi_conds_jensen_noscale": r"$\bar{I}({S'}; \theta_l^{S'}) + \breve{I}(X; Z_l^S | Y)$",

    "train_losses": "Train loss",
    "test_losses": "Test loss",
    "train_errs": "Train error",
    "test_errs": "Test error",

    "GE": "Generalization gap in error",
    "GL": "Generalization gap in loss",

    }

    pretty_summary = {
        "mean": "Mean",
        "max": "Max",
        "min": "Min",
        "fst": r"$l=1$",
        "last": r"$l=D$"
    }

    pretty_setting = {
        "model_name": "Model",
        "wd": "Weight decay",
        "batch_size": "Batch size",
        "seed": "Seed",
        "d_i": "Dataset instance"
    }

    for v_i, vname in enumerate(vnames_base):
        locals()["%s" % vname] = [] # num models: num layers

    arch_is = []
    details = []

    color_model_name = []
    color_wd = []
    color_batch_size = []

    color_seed = []
    color_d_i = []

    num_train = None
    compute_ind = 0
    for seed in range(num_seeds):
        for s_i, (batch_size, wd, model_name) in enumerate(settings):
            for d_i in data_instances:

                performance_path = os.path.join(performance_out_dir, "performance_%s.pt" % compute_ind)
                performance = torch.load(performance_path)

                if num_train is None: num_train = performance["num_train"]
                else: assert num_train == performance["num_train"]

                train_losses.append(torch.zeros(num_layers).fill_(performance["train_loss"]))
                test_losses.append(torch.zeros(num_layers).fill_(performance["test_loss"]))
                train_errs.append(torch.zeros(num_layers).fill_(performance["train_err"]))
                test_errs.append(torch.zeros(num_layers).fill_(performance["test_err"]))
                GL.append(torch.zeros(num_layers).fill_(performance["GL"]))
                GE.append(torch.zeros(num_layers).fill_(performance["GE"]))

                arch_is.append(model_names.index(model_name))

                for detail in ["model_name", "wd", "batch_size", "seed", "d_i"]:
                    locals()["color_%s" % detail].append(locals()[detail])

                details.append((batch_size, wd, model_name, seed, d_i))

                MI_X_Z_compute_ind_path = os.path.join(MI_X_Z_out_dir, "MI_X_Z_compute_ind_%s.pt" % compute_ind)
                MI_X_Z_results = torch.load(MI_X_Z_compute_ind_path)

                MI_jensens_curr = MI_X_Z_results["MI_Z_X_jensen"]
                MI_jensens.append(to_bits(MI_jensens_curr[:-1])) # number of layers

                MI_mcs_curr = MI_X_Z_results["MI_Z_X_mc"]
                MI_mcs.append(to_bits(MI_mcs_curr[:-1]))

                MI_cond_jensens_curr = MI_X_Z_results["avg_MI_Z_Xy_jensen"]
                MI_cond_jensens.append(to_bits(MI_cond_jensens_curr[:-1]))

                MI_cond_mcs_curr = MI_X_Z_results["avg_MI_Z_Xy_mc"]
                MI_cond_mcs.append(to_bits(MI_cond_mcs_curr[:-1]))

                theta_D_key_jensen_multiseed = "%s_%s_%s_%s" % (dataset_root, batch_size, wd, model_name)
                theta_D_key_jensen = "%s_%s_%s_%s_%s" % (dataset_root, seed, batch_size, wd, model_name)

                MI_theta_D_jensen = torch.load(os.path.join(args.out_dir, "MI_theta_D_jensen", "theta_D_key_%s.pt" % theta_D_key_jensen))
                MI_theta_D_jensen_multiseed = torch.load(os.path.join(args.out_dir, "MI_theta_D_jensen_multiseed", "theta_D_key_%s.pt" % theta_D_key_jensen_multiseed))

                if isinstance(MI_theta_D_jensen, tuple):
                    print("MI_theta_D_jensen tuple")
                    assert len(MI_theta_D_jensen) == 2
                    MI_theta_D_jensen = MI_theta_D_jensen[0]

                if isinstance(MI_theta_D_jensen_multiseed, tuple):
                    print("MI_theta_D_jensen_multiseed tuple")
                    assert len(MI_theta_D_jensen_multiseed) == 2
                    MI_theta_D_jensen_multiseed = MI_theta_D_jensen_multiseed[0]

                MI_theta_singles.append(to_bits(MI_theta_D_jensen[:-1]))
                MI_theta_multis.append(to_bits(MI_theta_D_jensen_multiseed[:-1]))

                MI_theta_singles_last.append(torch.zeros(num_layers).fill_(to_bits(MI_theta_D_jensen[-1])))
                MI_theta_multis_last.append(torch.zeros(num_layers).fill_(to_bits(MI_theta_D_jensen_multiseed[-1])))

                compute_ind += 1

    arch_is = torch.Tensor(arch_is)

    print("Collecting")
    for v_i, vname in enumerate(vnames_base):
        locals()["%s" % vname] =  torch.stack(locals()["%s" % vname], dim=0) # num models, num layers
        assert locals()["%s" % vname].shape[1] == num_layers
        num_models = locals()["%s" % vname].shape[0]

    combined_singles_mc_noscale = MI_theta_singles + MI_mcs
    combined_single_conds_mc_noscale = MI_theta_singles + MI_cond_mcs

    combined_singles_jensen_noscale = MI_theta_singles + MI_jensens
    combined_single_conds_jensen_noscale = MI_theta_singles + MI_cond_jensens

    combined_multis_mc_noscale = MI_theta_multis + MI_mcs
    combined_multi_conds_mc_noscale = MI_theta_multis + MI_cond_mcs

    combined_multis_jensen_noscale = MI_theta_multis + MI_jensens
    combined_multi_conds_jensen_noscale = MI_theta_multis + MI_cond_jensens

    # for each vname, generate actual metric from mean, min, max, fst, last collapsing across the layers
    for v_i, vname in enumerate(vnames):
        if vname in vnames_base_invariant:
            for summary in ["mean", "max", "min", "fst", "last"]:
                locals()["%s_%s" % (vname, summary)] = locals()["%s" % vname][:, 0]
        else:
            locals()["%s_mean" % vname] = locals()["%s" % vname].mean(dim=1)
            locals()["%s_max" % vname] = locals()["%s" % vname].max(dim=1)[0]

            if vname in ["MI_theta_singles", "MI_theta_multis"]:
                locals()["%s_min" % vname] = locals()["%s" % vname][:, 1:].min(dim=1)[0] # exclude first layer, which is empty
            else:
                locals()["%s_min" % vname] = locals()["%s" % vname].min(dim=1)[0]

            locals()["%s_fst" % vname] = locals()["%s" % vname][:, 0]
            locals()["%s_last" % vname] = locals()["%s" % vname][:, -1]

    sns.set_style("white")

    fig_all, ax = plt.subplots(1, figsize=(4, 4))
    fig_best, ax_best = plt.subplots(1, figsize=(4, 4))

    palette = itertools.cycle(sns.color_palette("deep"))
    best_model = test_losses[:, 0].argmin()
    print(details[best_model])
    layer_range = np.arange(1, num_layers + 1)

    for vname in vnames_perlayer_plot:
        res = locals()["%s" % vname] # num models, num layers

        # print example values
        res_str = ""
        res_best = res[best_model, :]
        for l in range(res_best.shape[0]):
            res_str += ("& %.4E" % res_best[l])
        print("%s \t%s \\\\" % (vnames_pretty[vname], res_str))

        # standardize the range across the layers
        res = res - res.min(dim=1)[0].unsqueeze(1)
        res = res / res.max(dim=1)[0].unsqueeze(1)

        mean = res.mean(dim=0) # average over models
        std = res.std(dim=0)

        color = next(palette)

        line, = ax.plot(layer_range, mean, label=vnames_pretty[vname], c=color, marker=".", linewidth=4)
        min_layer = mean.argmin()
        ax.fill_between(layer_range, mean - std, mean + std, color=color, alpha=0.15)
        ax_best.plot(layer_range, res[best_model], label=vnames_pretty[vname], c=color, marker="o", markerfacecolor=(color + (0.4,)))

    for a in [ax, ax_best]:
        a.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.25)) # bbox_to_anchor=(1, 0.5)
        a.set_xlabel("Layer index " + r"$l$")
        a.set_ylabel("Metric (normalized)")

    fig_all.savefig(os.path.join(args.out_dir, "per_layer_all.pdf"), bbox_inches="tight")
    fig_best.savefig(os.path.join(args.out_dir, "per_layer_best.pdf"), bbox_inches="tight")
    plt.close("all")

    # Compute correlations for each metric
    for v_i, vname in enumerate(vnames):
        for layer_summary in ["mean", "max", "min", "fst", "last"]: #  # , todo
            v_fullname = "%s_%s" % (vname, layer_summary)
            results_per_model = locals()[v_fullname]
            assert results_per_model.shape == (num_models,)

            if vname in vnames_base_invariant:
                diff = (locals()["%s_mean" % vname] - results_per_model).abs()
                if not (diff < eps).all():
                    print(vname)
                    print((diff - eps).abs().max())
                assert (diff < eps).all()

            r_GE, _ = stats.spearmanr(GE_mean, results_per_model)
            r_GL, _ = stats.spearmanr(GL_mean, results_per_model)
            p_GE, _ = stats.pearsonr(GE_mean, results_per_model)
            p_GL, _ = stats.pearsonr(GL_mean, results_per_model)
            tau_GE, _ = stats.kendalltau(GE_mean, results_per_model)
            tau_GL, _ = stats.kendalltau(GL_mean, results_per_model)

            pretty_name = clean(vname)
            if vname in vnames_pretty: pretty_name = vnames_pretty[vname]
            print("%s & %s & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f \\\\" % (pretty_name, pretty_summary[layer_summary], r_GL, r_GE, p_GL, p_GE, tau_GL, tau_GE))

            if layer_summary == "last":
                print("\\midrule")

            # plot scatter plot with trendline
            bare = False
            poly = 2
            if layer_summary == "min":
                for detail in ["model_name", "wd", "batch_size", "seed", "d_i"]:
                    if vname in vnames_base_invariant or detail == "model_name":
                        fig_per_metric, ax_per_metric = plt.subplots(1, figsize=(5, 5))
                        sns.set_style("dark")

                        df1 = create_dataframe(GL_mean, results_per_model, locals()["color_%s" % detail])
                        sns.scatterplot(data=df1, x="GL_curr", y="v", hue="arch", palette="deep", s=60, ax=ax_per_metric)

                        z2 = np.polyfit(GL_mean, results_per_model, poly)
                        x = np.linspace(GL_mean.min(), GL_mean.max(), 30)
                        res = np.polyval(z2, x)
                        if not bare: sns.lineplot(x=x, y=res, palette="deep", ax=ax_per_metric, linestyle="--")

                        ax_per_metric.set_ylim(ax_per_metric.get_ylim()[::-1])
                        assert ax_per_metric.get_ylim()[0] < ax_per_metric.get_ylim()[1]

                        ax_per_metric.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

                        if bare: ax_per_metric.legend(loc="upper right", title=pretty_setting[detail])
                        else: ax_per_metric.legend(loc="upper left")

                        name_curr = pretty_name
                        if vname not in vnames_base_invariant:
                            if not vname in ["MI_theta_singles", "MI_theta_multis"]:
                                name_curr = r"$\min_{l \in [D]}$ " + name_curr
                            else:
                                name_curr = r"$\min_{l \in [2 \dots D]}$ " + name_curr

                        ax_per_metric.set_ylabel(name_curr)

                        ax_per_metric.set_xlabel("Generalization gap in loss")
                        if not bare: ax_per_metric.set_title("Pearson correlation: %.3f" % (p_GL))
                        fig_per_metric.savefig(os.path.join(args.out_dir, "%s_%s.pdf" % (v_fullname, detail)), bbox_inches="tight")

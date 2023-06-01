import argparse
from toy.util.data import subsample_data_instance
from toy.util.general import set_seed, device, evaluate, clean, get_weight_norms
from toy.util.swag import *
from toy.util.model import StochasticMLP
import torch
import os
from datetime import datetime
from dnn.swag_repo.MI.util import to_bits, create_dataframe
import seaborn as sns

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import numpy as np

from collections import defaultdict
import scipy.stats as stats
import traceback



args_form = argparse.ArgumentParser(allow_abbrev=False)
args_form.add_argument("--inds_start", type=int, default=0)
args_form.add_argument("--inds_end", type=int, default=0)

args_form.add_argument("--lr", type=float, default=1e-2)
args_form.add_argument("--MI_const", type=float, default=1.5) #1.9
args_form.add_argument("--batch_sz", type=int, default=25) # 64
args_form.add_argument("--lamb_init", type=float, default=0.)

args_form.add_argument("--epochs", type=int, default=300)
args_form.add_argument("--swa_start", type=int, default=200)
args_form.add_argument("--out_dir", type=str, default="/network/scratch/x/xu.ji/mutual_info/v40") # v6c, 38, 39 (class even), 40 (more batches)

args_form.add_argument("--compute_MI_theta_D", default=False, action="store_true")
args_form.add_argument("--compute_MI_theta_D_only_start", type=int, default=0)
args_form.add_argument("--compute_MI_theta_D_only_end", type=int, default=0)

args_form.add_argument("--results", default=False, action="store_true")
args_form.add_argument("--data_only", default=False, action="store_true")

args = args_form.parse_args()
print(args)

print(args.out_dir)

data_instances = list(range(3))

archs = [[2, 256, 256, 128, 128, 5],
[2, 128, 128, 64, 64, 5],
[2, 64, 64, 32, 32, 5],
[2, 32, 32, 16, 16, 5],
]

decays = [0.0, 1e-2, 1e-1]

lamb_lrs = [0.0, 5e-4, 1e-3] # pre 41: [0.0, 5e-4, 1e-3], 41 [0.0, 5e-5, 1e-4]

seeds = [0, 1, 2]

subsamples = [0, 1, 2]


print("Num models:")
num_models = len(data_instances) * len(seeds) * len(lamb_lrs) * len(archs) * len(decays) * len(subsamples)
print(num_models)

plt.rc('axes', titlesize=15)
plt.rc('axes', labelsize=15)
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.rc('legend', fontsize=15)
plt.rc('font', size=15)


# s -> S
# \mathbf{S} -> S'
vnames_pretty = {

    # repr
    "MI_mcs": r"$\hat{I}(X; Z_l^S)$",
    "MI_cond_mcs": r"$\hat{I}(X; Z_l^S | Y)$",

    "MI_jensens": r"$\breve{I}(X; Z_l^S)$",
    "MI_cond_jensens": r"$\breve{I}(X; Z_l^S | Y)$",

    # model
    "MI_theta_singles": r"$\breve{I}({S'}; \theta_l^{S'})$", # jensen upper bound
    "MI_theta_singles_last": r"$\breve{I}({S'}; \theta_{D+1}^{S'})$",
    "MI_theta_orig": r"$\breve{I}({S'}; \theta_l^{S'})$ orig",

    # combined
    "combined_singles_mc": r"$\tilde{I}({S'}; \theta_l^{S'}) + \hat{I}(X; Z_l^S)$",
    "combined_single_conds_mc": r"$\tilde{I}({S'}; \theta_l^{S'}) + \hat{I}(X; Z_l^S | Y)$",

    "combined_singles_jensen": r"$\tilde{I}({S'}; \theta_l^{S'}) + \breve{I}(X; Z_l^S)$",
    "combined_single_conds_jensen": r"$\tilde{I}({S'}; \theta_l^{S'}) + \breve{I}(X; Z_l^S | Y)$",

    "combined_singles_mc_noscale": r"$\breve{I}({S'}; \theta_l^{S'}) + \hat{I}(X; Z_l^S)$",
    "combined_single_conds_mc_noscale": r"$\breve{I}({S'}; \theta_l^{S'}) + \hat{I}(X; Z_l^S | Y)$",

    "combined_singles_jensen_noscale": r"$\breve{I}({S'}; \theta_l^{S'}) + \breve{I}(X; Z_l^S)$",
    "combined_single_conds_jensen_noscale": r"$\breve{I}({S'}; \theta_l^{S'}) + \breve{I}(X; Z_l^S | Y)$",

    "num_params": r"Num. param. $m$",
    "VC": r"$m \log m$",
    "sum_weight_norms": r"$\sum_l \theta_l^\Sbb$",
    "prod_weight_norms": r"$\prod_l \theta_l^\Sbb$",

}


################################
# Training models
################################


model_ind = 0
for lamb_lr in lamb_lrs:
    for arch in archs:
        for decay in decays:

            for data_instance in data_instances:
                for subsample in subsamples:

                    for seed in seeds:

                        if model_ind in range(args.inds_start, args.inds_end): # exclusive
                            train_dl, test_dl = subsample_data_instance(args, args.batch_sz, data_instance, subsample)

                            print(("num batches", len(train_dl), len(test_dl)))

                            print("Doing model_ind %d, %s" % (model_ind, datetime.now()))
                            print((seed, lamb_lr, arch, decay, data_instance, subsample, seed))
                            sys.stdout.flush()

                            results_p = os.path.join(args.out_dir, "results_%d.pt" % model_ind)

                            if args.data_only or os.path.exists(results_p):
                                print("Skipping %s" % results_p)
                                model_ind += 1
                                continue

                            assert arch[-1] == args.C
                            set_seed(seed)

                            model = StochasticMLP(arch).to(device).train()
                            opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=decay)

                            model, swag_model, diagnostics = train_model_swag(model, arch, opt,
                                                                              train_dl, test_dl, args,
                                                                              lamb_lr)

                            train_acc, train_loss = evaluate(model, train_dl, args, "train", plot=True)
                            test_acc, test_loss = evaluate(model, test_dl, args, "test", plot=True)

                            print("train: %.3f %.3f, test: %.3f %.3f, MI: mc %s %s jensen %s %s, diagnostics: test %.3f %.3f swag %.3f %.3f" %
                                  (train_acc, train_loss, test_acc, test_loss,
                                   diagnostics["MI_mc"], diagnostics["MI_cond_mc"], diagnostics["MI_jensen"], diagnostics["MI_cond_jensen"],
                                   diagnostics["test_losses"][-1], diagnostics["test_accs"][-1],
                                   diagnostics["test_loss_swag"], diagnostics["test_acc_swag"],
                                   ))

                            # evaluate MI another time
                            MI_train_dl_5, _ = subsample_data_instance(args, args.batch_sz, data_instance, subsample, size=5, plot=False)

                            MI_mc_5 = est_MI(model, MI_train_dl_5.dataset,
                                                 sz=len(MI_train_dl_5.dataset), jensen=False,
                                                 requires_grad=False).item()
                            MI_cond_mc_5 = est_MI_cond(model, args.C, MI_train_dl_5,
                                                           sz=-1, jensen=False).item()

                            MI_jensen_5 = est_MI(model, MI_train_dl_5.dataset,
                                                     sz=len(MI_train_dl_5.dataset), jensen=True,
                                                     requires_grad=False).item()
                            MI_cond_jensen_5 = est_MI_cond(model, args.C, MI_train_dl_5,
                                                               sz=-1, jensen=True).item()


                            MI_train_dl_10, _ = subsample_data_instance(args, args.batch_sz, data_instance, subsample, size=10, plot=False)

                            MI_mc_10 = est_MI(model, MI_train_dl_10.dataset,
                                             sz=len(MI_train_dl_10.dataset), jensen=False,
                                             requires_grad=False).item()
                            MI_cond_mc_10 = est_MI_cond(model, args.C, MI_train_dl_10,
                                                       sz=-1, jensen=False).item()

                            MI_jensen_10 = est_MI(model, MI_train_dl_10.dataset,
                                                 sz=len(MI_train_dl_10.dataset), jensen=True,
                                                 requires_grad=False).item()
                            MI_cond_jensen_10 = est_MI_cond(model, args.C, MI_train_dl_10,
                                                           sz=-1, jensen=True).item()

                            # also other factors
                            weight_norms, num_params = get_weight_norms(model)

                            VC = num_params * np.log2(num_params)

                            sum_weight_norms = weight_norms.sum().item()

                            prod_weight_norms = weight_norms.prod().item()

                            nplots = len(diagnostics)
                            fig, axarr = plt.subplots(nplots, figsize=(4, nplots * 4))
                            for plot_i, (plot_name, plot_values) in enumerate(diagnostics.items()):

                                axarr[plot_i].plot(plot_values)
                                axarr[plot_i].set_ylabel(plot_name)
                                if plot_i == 0:
                                    axarr[plot_i].set_title("test acc %.3E loss %.3E"% (test_acc, test_loss))

                            plt.tight_layout()
                            fig.savefig(os.path.join(args.out_dir, "training_%d.pdf" % model_ind), bbox_inches="tight")
                            plt.close("all")

                            results = {
                                "model_ind": model_ind,

                                "swag_model": swag_model,
                                "model": model,

                                "arch": arch,
                                "decay": decay,
                                "lamb_lr": lamb_lr,
                                "seed": seed,

                                "data_instance": data_instance,
                                "subsample": subsample,

                                "train_acc": train_acc,
                                "train_loss": train_loss,
                                "test_acc": test_acc,
                                "test_loss": test_loss,

                                "gen_gap_err": train_acc - test_acc, # = 1 - test_acc - (1 - train_acc)
                                "gen_gap_loss": test_loss - train_loss,

                                "diagnostics": diagnostics
                            }

                            results["MI_mc_5"] = MI_mc_5
                            results["MI_cond_mc_5"] = MI_cond_mc_5
                            results["MI_jensen_5"] = MI_jensen_5
                            results["MI_cond_jensen_5"] = MI_cond_jensen_5

                            results["MI_mc_10"] = MI_mc_10
                            results["MI_cond_mc_10"] = MI_cond_mc_10
                            results["MI_jensen_10"] = MI_jensen_10
                            results["MI_cond_jensen_10"] = MI_cond_jensen_10

                            results["num_params"] = num_params
                            results["VC"] = VC
                            results["sum_weight_norms"] = sum_weight_norms
                            results["prod_weight_norms"] = prod_weight_norms

                            torch.save(results, results_p)

                        model_ind += 1


################################
# Model compression
################################

script_seed = 1

if args.compute_MI_theta_D:
    num_samples = 10 #5
    set_seed(script_seed) # for the script

    model_ind = 0
    setting_i = 0
    for lamb_lr in lamb_lrs:
        for arch in archs:
            for decay in decays:
                doing_setting = setting_i in list(range(args.compute_MI_theta_D_only_start, args.compute_MI_theta_D_only_end))

                #if doing_setting:
                #    swag_models = defaultdict(list) # seed, data instance, (subsample)
                swag_models = {}

                for data_instance in data_instances: #same order as above
                    for subsample in subsamples:
                        for seed in seeds:
                            if doing_setting:
                                results_f = os.path.join(args.out_dir, "results_%d.pt" % model_ind)
                                print("loading %s" % results_f)
                                results = torch.load(results_f)
                                if not seed in swag_models:
                                    swag_models[seed] = [[] for _ in range(len(data_instances))]
                                swag_models[seed][data_instance].append(results["swag_model"])

                            model_ind += 1 # increment even if not doing

                if doing_setting:
                    print("%s" % datetime.now())
                    sys.stdout.flush()

                    # theirs. One for the entire setting incl seed (across all datasets and subsamples) - average over data instances outside
                    for seed in seeds:
                        theta_D_key_single_seed = clean("%s_%s_%s_%s" % (seed, lamb_lr, arch, decay))

                        MI_theta_D_single_seed_all = []
                        for data_instance in data_instances:
                            print("inner %s" % datetime.now())
                            sys.stdout.flush()

                            # flat tensor 2 elements for each layer
                            MI_theta_D_single_seed = compute_MI_theta_D_single_seed_jensen(swag_models[seed][data_instance], num_samples, layers=[0, 1])
                            MI_theta_D_single_seed_all.append(MI_theta_D_single_seed)
                        MI_theta_D_single_seed_all = torch.stack(MI_theta_D_single_seed_all, dim=0)
                        MI_theta_D_single_seed_mean = MI_theta_D_single_seed_all.mean(dim=0)

                        torch.save({"MI_theta_D_single_seed_all": MI_theta_D_single_seed_all,
                                    "MI_theta_D_single_seed_mean": MI_theta_D_single_seed_mean},
                                   os.path.join(args.out_dir,"%d_theta_D_key_single_%s.pt" % (script_seed, theta_D_key_single_seed)))

                    print("ours full %s" % datetime.now())
                    sys.stdout.flush()
                    # Ours. 2 for the entire setting incl seed (across all datasets and subsamples)
                    for seed in seeds:
                        theta_D_key_orig_full = clean("%s_%s_%s_%s" % (seed, lamb_lr, arch, decay))
                        MI_theta_D_orig_full = compute_MI_theta_D_single_seed_jensen([item for sublist in swag_models[seed] for item in sublist], num_samples, layers=[0, 1])
                        torch.save(MI_theta_D_orig_full, os.path.join(args.out_dir, "%d_theta_D_key_orig_full_%s.pt" % (script_seed, theta_D_key_orig_full)))

                    print("ours %s" % datetime.now())
                    sys.stdout.flush()

                    # Ours. Just pick 1 subsample
                    for seed in seeds:
                        theta_D_key_orig = clean("%s_%s_%s_%s" % (seed, lamb_lr, arch, decay))
                        MI_theta_D_orig = compute_MI_theta_D_single_seed_jensen([swag_models[seed][di][0] for di in data_instances], num_samples, layers=[0, 1])
                        torch.save(MI_theta_D_orig, os.path.join(args.out_dir, "%d_theta_D_key_orig_%s.pt" % (script_seed, theta_D_key_orig)))


                    print(("MI results for %s (%s): single %s (%s), orig full %s, orig %s" % (setting_i,
                                                                                          str((lamb_lr, arch, decay)),

                                                                                          MI_theta_D_single_seed_mean,
                                                                                          MI_theta_D_single_seed_all,

                                                                                          MI_theta_D_orig_full,
                                                                                          MI_theta_D_orig
                                                                                          )))

                setting_i += 1


################################
# Results
################################

use_orig = False # true
suff_base = ""
if not use_orig: suff_base = "_10"

all_gaps = True # true

if args.results:
    sns.set_style("dark")  # darkgrid, white grid, dark, white and ticks

    plot = True
    print_summary = False

    vnames_base_invariant = [
        "MI_theta_singles_last",
    ]

    vnames_base_extra = [
        "num_params",
        "VC",
        "sum_weight_norms",
        "prod_weight_norms"
    ]

    vnames_base = ["MI_mcs", "MI_cond_mcs", "MI_jensens", "MI_cond_jensens",
                   "MI_theta_singles", "MI_theta_orig_full", "MI_theta_orig"] + vnames_base_invariant + vnames_base_extra

    vnames = vnames_base + [
        "combined_singles_mc", "combined_single_conds_mc",
        "combined_singles_jensen", "combined_single_conds_jensen",

        "combined_singles_mc_noscale", "combined_single_conds_mc_noscale",
        "combined_singles_jensen_noscale", "combined_single_conds_jensen_noscale",



        "combined_orig_full_mc", "combined_orig_full_conds_mc",
        "combined_orig_full_jensen", "combined_orig_full_conds_jensen",

        "combined_orig_full_mc_noscale", "combined_orig_full_conds_mc_noscale",
        "combined_orig_full_jensen_noscale", "combined_orig_full_conds_jensen_noscale",



        "combined_orig_mc", "combined_orig_conds_mc",
        "combined_orig_jensen", "combined_orig_conds_jensen",

        "combined_orig_mc_noscale", "combined_orig_conds_mc_noscale",
        "combined_orig_jensen_noscale", "combined_orig_conds_jensen_noscale",

        # others
        #"combined_multis_mc", "combined_multi_conds_mc",
        #"combined_multis_jensen", "combined_multi_conds_jensen",

        #"combined_multis_mc_noscale", "combined_multi_conds_mc_noscale",
        #"combined_multis_jensen_noscale", "combined_multi_conds_jensen_noscale",
    ]


    thresh = 0.85
    #thresh_loss = 0.1
    poly = 2

    for to_vary in ["none"]:
        print("Varying: %s" % to_vary)

        for lamb_lr_i, lamb_lr_curr in enumerate(lamb_lrs):
            details = defaultdict(list)

            if not print_summary: print("\nDoing lamb_lr %s" % lamb_lr_curr)

            GL = defaultdict(list)

            train_losses = defaultdict(list)
            train_accs = defaultdict(list)
            test_losses = defaultdict(list)
            test_accs = defaultdict(list)

            for v_i, vname in enumerate(vnames_base):
                locals()["%s" % vname] = defaultdict(list)

            counted = 0
            skipped1 = 0
            skipped2 = 0
            skipped3 = 0
            model_ind = 0

            metrics = ["tau_GL", "r_GL", "p_GL"]

            keys = {}
            for lamb_lr in lamb_lrs:
                for arch_i, arch in enumerate(archs):
                    for decay_i, decay in enumerate(decays):

                        for data_i, data_instance in enumerate(data_instances):
                            for seed_i, seed in enumerate(seeds):

                                if lamb_lr == lamb_lr_curr:

                                    try:
                                        r = torch.load(os.path.join(args.out_dir, "results_%d.pt" % model_ind))

                                        if r["train_acc"] > thresh: # (r["train_acc"] > r["test_acc"]) and

                                            if all_gaps or ((not all_gaps) and r["gen_gap_loss"] > 0.):
                                                key = get_key(arch_i, decay_i, data_i, seed_i, to_vary)
                                                keys[key] = 1

                                                GL[key].append(r["gen_gap_loss"])

                                                train_losses[key].append(r["train_loss"])
                                                train_accs[key].append(r["train_acc"])
                                                test_losses[key].append(r["test_loss"])
                                                test_accs[key].append(r["test_acc"])

                                                if not use_orig:
                                                    MI_mcs[key].append(to_bits(r["MI_mc" + suff_base]))
                                                    MI_cond_mcs[key].append(to_bits(r["MI_cond_mc" + suff_base]))
                                                    MI_jensens[key].append(to_bits(r["MI_jensen" + suff_base]))
                                                    MI_cond_jensens[key].append(to_bits(r["MI_cond_jensen" + suff_base]))
                                                else:
                                                    MI_mcs[key].append(to_bits(r["diagnostics"]["MI_mc" + suff_base]))
                                                    MI_cond_mcs[key].append(to_bits(r["diagnostics"]["MI_cond_mc" + suff_base]))
                                                    MI_jensens[key].append(to_bits(r["diagnostics"]["MI_jensen" + suff_base]))
                                                    MI_cond_jensens[key].append(to_bits(r["diagnostics"]["MI_cond_jensen" + suff_base]))

                                                # model compression
                                                theta_D_key_for_seed = clean("%s_%s_%s_%s" % (seed, lamb_lr, arch, decay))
                                                theta_single = to_bits(torch.load(os.path.join(args.out_dir, "%d_theta_D_key_single_%s.pt" % (script_seed, theta_D_key_for_seed)))["MI_theta_D_single_seed_mean"])

                                                # factor
                                                #MI_theta_singles[key].append(theta_single[0].item())
                                                #MI_theta_singles_last[key].append(theta_single[1].item())

                                                # exact
                                                MI_theta_singles[key].append(np.sqrt((2./ 50.) * theta_single[0].item()))
                                                MI_theta_singles_last[key].append(np.sqrt((2./ 50.) * theta_single[1].item()))


                                                theta_orig_full = to_bits(torch.load(os.path.join(args.out_dir, "%d_theta_D_key_orig_full_%s.pt" % (script_seed, theta_D_key_for_seed))))
                                                MI_theta_orig_full[key].append(theta_orig_full[0].item())

                                                theta_orig = to_bits(torch.load(os.path.join(args.out_dir, "%d_theta_D_key_orig_%s.pt" % (script_seed, theta_D_key_for_seed))))
                                                MI_theta_orig[key].append(theta_orig[0].item())


                                                num_params[key].append(r["num_params"])
                                                VC[key].append(r["VC"])
                                                sum_weight_norms[key].append(r["sum_weight_norms"])
                                                prod_weight_norms[key].append(r["prod_weight_norms"])

                                                details[key].append(arch_name(arch))
                                            else:
                                                skipped3 += 1

                                        else:
                                            skipped2 += 1

                                        counted += 1
                                    except Exception as e:
                                        print(traceback.format_exc())
                                        skipped1 += 1

                                model_ind += 1

            print("Counts %s, skipped err %s, thresh %s, gen gap %s" % (counted, skipped1, skipped2, skipped3))

            for metric in metrics:
                locals()["results_%s" % metric] = defaultdict(dict)

            for key in keys:
                GL_curr = np.array(GL[key])

                archs_key = details[key]

                train_losses_curr = np.array(train_losses[key])
                train_accs_curr = np.array(train_accs[key])
                test_losses_curr = np.array(test_losses[key])
                test_accs_curr = np.array(test_accs[key])

                MI_mcs_curr = (np.array(MI_mcs[key])) # all models for key
                MI_cond_mcs_curr = (np.array(MI_cond_mcs[key]))

                MI_jensens_curr = (np.array(MI_jensens[key]))
                MI_cond_jensens_curr = (np.array(MI_cond_jensens[key]))

                # model compression
                MI_theta_singles_curr = (np.array(MI_theta_singles[key])) # each one combined with the above 4
                MI_theta_singles_last_curr = (np.array(MI_theta_singles_last[key]))  # each one combined with the above 4

                MI_theta_orig_full_curr = (np.array(MI_theta_orig_full[key]))
                MI_theta_orig_curr = (np.array(MI_theta_orig[key]))

                num_params_curr = np.array(num_params[key])
                VC_curr = np.array(VC[key])
                sum_weight_norms_curr = np.array(sum_weight_norms[key])
                prod_weight_norms_curr = np.array(prod_weight_norms[key])

                scales = [[], [], []]
                for vname in ["MI_mcs_curr", "MI_cond_mcs_curr", "MI_jensens_curr", "MI_cond_jensens_curr"]:
                    v = locals()[vname]
                    scales[0].append(v.mean() / MI_theta_singles_curr.mean())
                    scales[1].append(v.mean() / MI_theta_orig_full_curr.mean())
                    scales[2].append(v.mean() / MI_theta_orig_curr.mean())

                print("Scales for %s %s" % (lamb_lr_curr, lamb_lr_i))
                print(scales)

                combined_singles_mc_curr = scales[0][0] * MI_theta_singles_curr + MI_mcs_curr
                combined_single_conds_mc_curr = scales[0][1] * MI_theta_singles_curr + MI_cond_mcs_curr
                combined_singles_jensen_curr = scales[0][2] * MI_theta_singles_curr + MI_jensens_curr
                combined_single_conds_jensen_curr = scales[0][3] * MI_theta_singles_curr + MI_cond_jensens_curr

                combined_orig_full_mc_curr = scales[1][0] * MI_theta_orig_full_curr + MI_mcs_curr
                combined_orig_full_conds_mc_curr = scales[1][1] * MI_theta_orig_full_curr + MI_cond_mcs_curr
                combined_orig_full_jensen_curr = scales[1][2] * MI_theta_orig_full_curr + MI_jensens_curr
                combined_orig_full_conds_jensen_curr = scales[1][3] * MI_theta_orig_full_curr + MI_cond_jensens_curr

                combined_orig_mc_curr = scales[2][0] * MI_theta_orig_curr + MI_mcs_curr
                combined_orig_conds_mc_curr = scales[2][1] * MI_theta_orig_curr + MI_cond_mcs_curr
                combined_orig_jensen_curr = scales[2][2] * MI_theta_orig_curr + MI_jensens_curr
                combined_orig_conds_jensen_curr = scales[2][3] * MI_theta_orig_curr + MI_cond_jensens_curr


                # noscale


                combined_singles_mc_noscale_curr = MI_theta_singles_curr + MI_mcs_curr
                combined_single_conds_mc_noscale_curr = MI_theta_singles_curr + MI_cond_mcs_curr
                combined_singles_jensen_noscale_curr = MI_theta_singles_curr + MI_jensens_curr
                combined_single_conds_jensen_noscale_curr = MI_theta_singles_curr + MI_cond_jensens_curr

                combined_orig_full_mc_noscale_curr = MI_theta_orig_full_curr + MI_mcs_curr
                combined_orig_full_conds_mc_noscale_curr = MI_theta_orig_full_curr + MI_cond_mcs_curr
                combined_orig_full_jensen_noscale_curr = MI_theta_orig_full_curr + MI_jensens_curr
                combined_orig_full_conds_jensen_noscale_curr = MI_theta_orig_full_curr + MI_cond_jensens_curr

                combined_orig_mc_noscale_curr = MI_theta_orig_curr + MI_mcs_curr
                combined_orig_conds_mc_noscale_curr = MI_theta_orig_curr + MI_cond_mcs_curr
                combined_orig_jensen_noscale_curr = MI_theta_orig_curr + MI_jensens_curr
                combined_orig_conds_jensen_noscale_curr = MI_theta_orig_curr + MI_cond_jensens_curr


                if plot:
                    fig, axarr = plt.subplots(len(vnames), 2, figsize=(2 * 3, len(vnames) * 3)) # all
                    fig2, axarr2 = plt.subplots(1, figsize=(5, 5)) # MI_mcs
                    fig3, axarr3 = plt.subplots(1, figsize=(5, 5)) # combined_single_conds_mc
                    fig4, axarr4 = plt.subplots(1, figsize=(5, 5)) # combined_single_conds_mc
                    fig5, axarr5 = plt.subplots(1, figsize=(5, 5)) # combined_single_conds_mc

                for v_i, vname in enumerate(vnames):
                    v = locals()["%s_curr" % vname]

                    for metric in metrics: # wht
                        locals()["%s_curr" % metric] = np.nan

                    try:
                        tau_GL_curr, _ = stats.kendalltau(GL_curr, v)
                        r_GL_curr, _ = stats.spearmanr(GL_curr, v)
                        p_GL_curr, _ = stats.pearsonr(GL_curr, v)
                    except Exception as e:
                        print(e)
                        continue

                    for metric in metrics:
                        locals()["results_%s" % metric][vname][key] = locals()["%s_curr" % metric]

                    if not print_summary:
                        pretty_name = vname
                        if vname in vnames_pretty: pretty_name = vnames_pretty[vname]
                        print("%s \t& %.4f & %.4f & %.4f \\\\" % (pretty_name, r_GL_curr, p_GL_curr, tau_GL_curr))

                    if plot:

                        # all
                        axarr[v_i, 0].scatter(GL_curr, v)

                        z0 = np.polyfit(GL_curr, v, poly)
                        x = np.linspace(GL_curr.min(), GL_curr.max(), 30)
                        axarr[v_i, 0].plot(x, np.polyval(z0, x), "r--")

                        axarr[v_i, 0].set_ylabel(vname)

                        axarr[v_i, 0].set_xlabel("GL")

                        axarr[v_i, 0].set_title("tau %.3f, r %.3f, \n p %s" % (tau_GL_curr, r_GL_curr, p_GL_curr))

                        # individuals
                        if vname == "MI_cond_jensens":
                            # axarr2.scatter(GL_curr, v)

                            df1 = create_dataframe(GL_curr, v, archs_key)
                            sns.scatterplot(data=df1, x="GL_curr", y="v", hue="arch", palette="deep", s=60, ax=axarr2)

                            z2 = np.polyfit(GL_curr, v, poly)
                            x = np.linspace(GL_curr.min(), GL_curr.max(), 30)
                            res = np.polyval(z2, x)
                            assert res.shape == x.shape

                            #axarr2.plot(x, res, "r--")
                            sns.lineplot(x=x, y=res, palette="deep", ax=axarr2, linestyle="--")

                            if lamb_lr_i == 0:
                                axarr2.legend(title="Width", loc="lower right")
                            else:
                                axarr2.legend(title="Width", loc="upper left")

                            axarr2.set_ylabel(vnames_pretty[vname])
                            axarr2.set_xlabel("Generalization gap in loss")
                            axarr2.set_title("Pearson correlation: %.3f" % (p_GL_curr))
                            #axarr2.set_xticklabels(axarr2.get_xticklabels(), rotation=45)

                        if vname == "MI_theta_singles":
                            # axarr2.scatter(GL_curr, v)

                            df1 = create_dataframe(GL_curr, v, archs_key)
                            sns.scatterplot(data=df1, x="GL_curr", y="v", hue="arch", palette="deep", s=60, ax=axarr3)

                            z2 = np.polyfit(GL_curr, v, poly)
                            x = np.linspace(GL_curr.min(), GL_curr.max(), 30)
                            res = np.polyval(z2, x)
                            assert res.shape == x.shape

                            #axarr2.plot(x, res, "r--")
                            sns.lineplot(x=x, y=res, palette="deep", ax=axarr3, linestyle="--")

                            if lamb_lr_i == 0:
                                axarr3.legend(title="Width", loc="lower right")
                            else:
                                axarr3.legend(title="Width", loc="upper left")

                            axarr3.set_ylabel(vnames_pretty[vname])
                            axarr3.set_xlabel("Generalization gap in loss")
                            axarr3.set_title("Pearson correlation: %.3f" % (p_GL_curr))
                            #axarr2.set_xticklabels(axarr2.get_xticklabels(), rotation=45)

                        if vname == "MI_theta_singles_last":
                            #axarr3.scatter(GL_curr, v)
                            df2 = create_dataframe(GL_curr, v, archs_key)
                            sns.scatterplot(data=df2, x="GL_curr", y="v",
                                            hue="arch", palette="deep", s=60, ax=axarr4)

                            z3 = np.polyfit(GL_curr, v, poly)
                            x = np.linspace(GL_curr.min(), GL_curr.max(), 30)
                            res = np.polyval(z3, x)
                            assert res.shape == x.shape

                            sns.lineplot(x=x, y=res, palette="deep", ax=axarr4, linestyle="--")

                            if lamb_lr_i == 0:
                                axarr4.legend(title="Width", loc="lower right")
                            else:
                                axarr4.legend(title="Width", loc="upper left")

                            axarr4.set_ylabel(vnames_pretty[vname])
                            axarr4.set_xlabel("Generalization gap in loss")
                            axarr4.set_title("Pearson correlation: %.3f" % (p_GL_curr))

                        if vname == "MI_theta_orig":
                            df2 = create_dataframe(GL_curr, v, archs_key)
                            sns.scatterplot(data=df2, x="GL_curr", y="v",
                                            hue="arch", palette="deep", s=60, ax=axarr5)

                            z3 = np.polyfit(GL_curr, v, poly)
                            x = np.linspace(GL_curr.min(), GL_curr.max(), 30)
                            res = np.polyval(z3, x)
                            assert res.shape == x.shape

                            sns.lineplot(x=x, y=res, palette="deep", ax=axarr5, linestyle="--")

                            if lamb_lr_i == 0:
                                axarr5.legend(title="Width", loc="lower right")
                            else:
                                axarr5.legend(title="Width", loc="upper left")

                            axarr5.set_ylabel(vnames_pretty[vname])
                            axarr5.set_xlabel("Generalization gap in loss")
                            axarr5.set_title("Pearson correlation: %.3f" % (p_GL_curr))


                if plot:
                    plt.tight_layout()
                    fig.savefig(os.path.join(args.out_dir, "summary_lr_%s_key_%s.pdf" % (lamb_lr_curr, key)), bbox_inches="tight")
                    fig2.savefig(os.path.join(args.out_dir, "summary_lr_%s_key_%s_MI_cond_jensens.pdf" % (lamb_lr_curr, key)), bbox_inches="tight")
                    fig3.savefig(os.path.join(args.out_dir, "summary_lr_%s_key_%s_MI_theta_singles.pdf" % (lamb_lr_curr, key)), bbox_inches="tight")
                    fig4.savefig(os.path.join(args.out_dir, "summary_lr_%s_key_%s_MI_theta_singles_last.pdf" % (lamb_lr_curr, key)), bbox_inches="tight")
                    fig5.savefig(os.path.join(args.out_dir, "summary_lr_%s_key_%s_MI_theta_orig.pdf" % (lamb_lr_curr, key)), bbox_inches="tight")

                    plt.close("all")

            for metric in metrics:
                locals()["repr_best_%s" % metric] = (-np.inf, None)
                locals()["model_incl_best_%s" % metric] = (-np.inf, None)

            for v_i, vname in enumerate(vnames):
                if vname in ["MI_mcs", "MI_cond_mcs", "MI_jensens", "MI_cond_jensens"]:
                    pref = "repr"
                else:
                    pref = "model_incl"

                for metric in metrics:
                    avg_results = np.array(list(locals()["results_%s" % metric][vname].values())).mean()
                    if avg_results > locals()["%s_best_%s" % (pref, metric)][0]:
                        locals()["%s_best_%s" % (pref, metric)] = (avg_results, vname)

            if print_summary:
                print("Results for %s, %s" % (lamb_lr_curr, to_vary))
                for metric in metrics:
                    print("%s: %s" % ("repr_best_%s" % metric, locals()["repr_best_%s" % metric]))
                    print("%s: %s" % ("model_incl_best_%s" % metric, locals()["model_incl_best_%s" % metric]))

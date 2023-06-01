import argparse
from toy.util.data import basic_data_instance
from toy.util.general import set_seed, device, evaluate, clean, get_weight_norms, compute_factors_binning
from toy.util.swag import *
from toy.util.model import BasicMLP
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

args_form = argparse.ArgumentParser(allow_abbrev=False)
args_form.add_argument("--inds_start", type=int, default=0)
args_form.add_argument("--inds_end", type=int, default=0)

args_form.add_argument("--lr", type=float, default=1e-2)
args_form.add_argument("--batch_sz", type=int, default=25)
args_form.add_argument("--lamb_init", type=float, default=0.)

args_form.add_argument("--epochs", type=int, default=200)
args_form.add_argument("--swa_start", type=int, default=150)
args_form.add_argument("--out_dir_root", type=str, default="/network/scratch/x/xu.ji/mutual_info")
args_form.add_argument("--version", type=int, default=37)

args_form.add_argument("--compute_MI_theta_D", default=False, action="store_true")
args_form.add_argument("--compute_MI_theta_D_only_start", type=int, default=0)
args_form.add_argument("--compute_MI_theta_D_only_end", type=int, default=0)

args_form.add_argument("--results", default=False, action="store_true")
args_form.add_argument("--data_only", default=False, action="store_true")

args_form.add_argument("--MI_reg_layers", type=int, nargs="+", default=[3])

args_form.add_argument("--num_bins", type=int, default=10)
args_form.add_argument("--num_bins_cond", type=int, default=10)

args = args_form.parse_args()
version = args.version
args.out_dir = os.path.join(args.out_dir_root, "v" + str(version))
print(args)

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

data_instances = list(range(3))

decays = [0.0, 1e-3, 1e-2]

seeds = [0, 1, 2]

lamb_lrs = [0.0, 1.0]

archs = [
[2, 512, 512, 256, 256, 5],
[2, 256, 256, 128, 128, 5],
[2, 128, 128, 64, 64, 5],
]

print("Num models:")
num_models = len(data_instances) * len(seeds) * len(lamb_lrs) * len(archs) * len(decays)
print(num_models)

plt.rc('axes', titlesize=7)
plt.rc('axes', labelsize=7)
plt.rc('xtick', labelsize=7)
plt.rc('ytick', labelsize=7)
plt.rc('legend', fontsize=7)
plt.rc('font', size=7)

vnames_pretty = {

    # repr
    "MI_mcs": r"$\hat{I}(X; Z_l^s)$",
    "MI_cond_mcs": r"$\hat{I}(X; Z_l^s | Y)$",

    "MI_jensens": r"$\breve{I}(X; Z_l^s)$",
    "MI_cond_jensens": r"$\breve{I}(X; Z_l^s | Y)$",

    # model
    "MI_theta_singles": r"$\breve{I}(\mathbf{S}; \theta_l^\mathbf{S})$", # jensen upper bound
    "MI_theta_multis": r"$\bar{I}(\mathbf{S}; \theta_l^\mathbf{S})$", # double jensen bound

    # combined
    "combined_singles_mc": r"$\tilde{I}(\mathbf{S}; \theta_l^\mathbf{S}) + \hat{I}(X; Z_l^s)$",
    "combined_single_conds_mc": r"$\tilde{I}(\mathbf{S}; \theta_l^\mathbf{S}) + \hat{I}(X; Z_l^s | Y)$",

    "combined_singles_jensen": r"$\tilde{I}(\mathbf{S}; \theta_l^\mathbf{S}) + \breve{I}(X; Z_l^s)$",
    "combined_single_conds_jensen": r"$\tilde{I}(\mathbf{S}; \theta_l^\mathbf{S}) + \breve{I}(X; Z_l^s | Y)$",

    "combined_singles_mc_noscale": r"$\breve{I}(\mathbf{S}; \theta_l^\mathbf{S}) + \hat{I}(X; Z_l^s)$",
    "combined_single_conds_mc_noscale": r"$\breve{I}(\mathbf{S}; \theta_l^\mathbf{S}) + \hat{I}(X; Z_l^s | Y)$",

    "combined_singles_jensen_noscale": r"$\breve{I}(\mathbf{S}; \theta_l^\mathbf{S}) + \breve{I}(X; Z_l^s)$",
    "combined_single_conds_jensen_noscale": r"$\breve{I}(\mathbf{S}; \theta_l^\mathbf{S}) + \breve{I}(X; Z_l^s | Y)$",

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
                for seed in seeds:

                    if model_ind in range(args.inds_start, args.inds_end): # exclusive
                        train_dl, test_dl = basic_data_instance(args, args.batch_sz, data_instance)

                        print(("num batches", len(train_dl), len(test_dl)))

                        print("Doing model_ind %d, %s" % (model_ind, datetime.now()))
                        print((seed, lamb_lr, arch, decay))
                        sys.stdout.flush()

                        results_p = os.path.join(args.out_dir, "results_%d.pt" % model_ind)

                        if args.data_only:
                            print("Skipping %s" % results_p)
                            model_ind += 1
                            continue

                        assert arch[-1] == args.C
                        set_seed(seed)

                        model = BasicMLP(arch).to(device).train()
                        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=decay)

                        model, swag_model, diagnostics = train_model_swag_binning(model, arch, opt, train_dl, test_dl, args, lamb_lr=lamb_lr)

                        train_acc, train_loss = evaluate(model, train_dl, args, "train", plot=True)
                        test_acc, test_loss = evaluate(model, test_dl, args, "test", plot=True)

                        print("train: %.3f %.3f, test: %.3f %.3f, MI: mc %s, diagnostics: test %.3f %.3f swag %.3f %.3f" %
                              (train_acc, train_loss, test_acc, test_loss,
                               diagnostics["MI_mc_perlayer"],
                               diagnostics["test_losses"][-1], diagnostics["test_accs"][-1],
                               diagnostics["test_loss_swag"], diagnostics["test_acc_swag"],
                               ))

                        # evaluate MI another time
                        MI_train_dl_5, _ = basic_data_instance(args, args.batch_sz, data_instance, size=5)

                        MI_mc_5_perlayer, ranges_5, bin_szs_5 = est_MI_binning(model, MI_train_dl_5.dataset, num_bins=args.num_bins)

                        MI_cond_mc_5_perlayer = est_MI_binning_cond(model, MI_train_dl_5.dataset, args.C, num_bins_cond=args.num_bins_cond)

                        MI_train_dl_10, _ = basic_data_instance(args, args.batch_sz, data_instance, size=10)

                        MI_mc_10_perlayer, ranges_10, bin_szs_10 = est_MI_binning(model, MI_train_dl_10.dataset, num_bins=args.num_bins)

                        MI_cond_mc_10_perlayer = est_MI_binning_cond(model, MI_train_dl_10.dataset, args.C, num_bins_cond=args.num_bins_cond)

                        print(("other MIs:", MI_mc_5_perlayer, MI_mc_10_perlayer))

                        # other factors
                        weight_norms, num_params = get_weight_norms(model)

                        VC = num_params * np.log2(num_params)

                        sum_weight_norms = weight_norms.sum().item()

                        prod_weight_norms = weight_norms.prod().item()

                        nplots = len(diagnostics)
                        fig, axarr = plt.subplots(nplots, figsize=(4, nplots * 4))
                        for plot_i, (plot_name, plot_values) in enumerate(diagnostics.items()):
                            if (not (isinstance(plot_values, list) or isinstance(plot_values, torch.Tensor))) \
                                or (not (isinstance(plot_values[0], list) or isinstance(plot_values[0], torch.Tensor))):
                                print(plot_name, plot_values.__class__)
                                axarr[plot_i].plot(plot_values)
                                axarr[plot_i].set_ylabel(plot_name)
                                if plot_i == 0:
                                    axarr[plot_i].set_title("test acc %.3E loss %.3E"% (test_acc, test_loss))
                            else:
                                print("skipping plot %s" % plot_name)
                                print(plot_values.__class__, plot_values[0].__class__, plot_values[0])

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

                            "train_acc": train_acc,
                            "train_loss": train_loss,
                            "test_acc": test_acc,
                            "test_loss": test_loss,

                            "gen_gap_err": train_acc - test_acc, # = 1 - test_acc - (1 - train_acc)
                            "gen_gap_loss": test_loss - train_loss,

                            "diagnostics": diagnostics
                        }

                        results["MI_mc_5_perlayer"] = MI_mc_5_perlayer
                        results["MI_cond_mc_5_perlayer"] = MI_cond_mc_5_perlayer
                        results["ranges_5"] = ranges_5
                        results["bin_szs_5"] = bin_szs_5

                        results["MI_mc_10_perlayer"] = MI_mc_10_perlayer
                        results["MI_cond_mc_10_perlayer"] = MI_cond_mc_10_perlayer
                        results["ranges_10"] = ranges_10
                        results["bin_szs_10"] = bin_szs_10

                        results["num_params"] = num_params
                        results["VC"] = VC
                        results["sum_weight_norms"] = sum_weight_norms
                        results["prod_weight_norms"] = prod_weight_norms

                        torch.save(results, results_p)

                    model_ind += 1


################################
# Model compression
################################

if args.compute_MI_theta_D:
    num_samples = 10
    set_seed(1)

    model_ind = 0
    setting_i = 0
    for lamb_lr in lamb_lrs:
        for arch in archs:
            for decay in decays:
                doing_setting = setting_i in list(range(args.compute_MI_theta_D_only_start, args.compute_MI_theta_D_only_end))

                if doing_setting: swag_models = defaultdict(list)
                for data_instance in data_instances:
                    for seed in seeds:
                        if doing_setting:
                            results_f = os.path.join(args.out_dir, "results_%d.pt" % model_ind)
                            print("loading %s" % results_f)
                            results = torch.load(results_f)
                            swag_models[seed].append(results["swag_model"])

                        model_ind += 1 # increment even if not doing

                if doing_setting:
                    print("%s" % datetime.now())
                    sys.stdout.flush()

                    for seed in seeds:
                        theta_D_key_single_seed = clean("%s_%s_%s_%s" % (seed, lamb_lr, arch, decay))

                        MI_theta_D_single_seed = compute_MI_theta_D_single_seed_jensen(swag_models[seed], num_samples, layers=list(range(len(arch) - 1)))
                        torch.save(MI_theta_D_single_seed, os.path.join(args.out_dir,"%d_theta_D_key_single_%s.pt" % (1, theta_D_key_single_seed)))

                    theta_D_key_multi_seed = clean("%s_%s_%s" % (lamb_lr, arch, decay))
                    MI_theta_D_multi_seed = compute_MI_theta_D_multiseed_jensen(swag_models, num_samples, layers=list(range((len(arch) - 1))))
                    torch.save(MI_theta_D_multi_seed, os.path.join(args.out_dir, "%d_theta_D_key_multi_%s.pt" % (1, theta_D_key_multi_seed)))

                    print(("MI results for %s (%s): single %s, multi %s" % (setting_i, theta_D_key_multi_seed, MI_theta_D_single_seed, MI_theta_D_multi_seed)))

                setting_i += 1


################################
# Results
################################

use_orig = False # use training set (true) or larger sample of data (false)
suff_base = ""
if not use_orig: suff_base = "_5"
all_gaps = True

if args.results:
    sns.set_style("dark")

    plot = True
    print_summary = False

    vnames_base_invariant = [
    ]

    vnames_base_extra = [
        "num_params",
        "VC",
        "sum_weight_norms",
        "prod_weight_norms"
    ]

    vnames_base = ["MI_mcs", "MI_cond_mcs",
                   "MI_theta_singles", "MI_theta_multis"] + vnames_base_invariant + vnames_base_extra

    vnames = vnames_base + [
        "combined_singles_mc", "combined_single_conds_mc",

        "combined_singles_mc_noscale", "combined_single_conds_mc_noscale",

        "combined_multis_mc", "combined_multi_conds_mc",

        "combined_multis_mc_noscale", "combined_multi_conds_mc_noscale",
    ]

    focus_vnames = [
        "MI_mcs", "MI_cond_mcs",
        "MI_theta_singles", "MI_theta_multis",

        "combined_singles_mc", "combined_single_conds_mc",
        "combined_multis_mc", "combined_multi_conds_mc",
    ]

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

                                    r = torch.load(os.path.join(args.out_dir, "results_%d.pt" % model_ind))
                                    if all_gaps or ((not all_gaps) and r["gen_gap_loss"] > 0.):
                                        key = get_key(arch_i, decay_i, data_i, seed_i, to_vary)
                                        keys[key] = 1

                                        GL[key].append(r["gen_gap_loss"])

                                        train_losses[key].append(r["train_loss"])
                                        train_accs[key].append(r["train_acc"])
                                        test_losses[key].append(r["test_loss"])
                                        test_accs[key].append(r["test_acc"])

                                        if not use_orig:
                                            MI_mcs_raw = to_bits(np.array(r["MI_mc" + suff_base + "_perlayer"]))
                                            conds_raw = to_bits(np.array(r["MI_cond_mc" + suff_base + "_perlayer"]).mean(axis=0))
                                        else:
                                            MI_mcs_raw = to_bits(np.array(r["diagnostics"]["MI_mc" + suff_base + "_perlayer"]))
                                            conds_raw = to_bits(np.array(r["diagnostics"]["MI_cond_mc" + suff_base + "_perlayer"]).mean(axis=0))

                                        MI_mcs[key].append(MI_mcs_raw)
                                        MI_cond_mcs[key].append(conds_raw)

                                        theta_D_key_single_seed = clean("%s_%s_%s_%s" % (seed, lamb_lr, arch, decay))
                                        theta_single = to_bits(torch.load(os.path.join(args.out_dir, "%d_theta_D_key_single_%s.pt" % (1, theta_D_key_single_seed))).numpy())

                                        theta_D_key_multi_seed = clean("%s_%s_%s" % (lamb_lr, arch, decay))
                                        theta_multi = to_bits(torch.load(os.path.join(args.out_dir, "%d_theta_D_key_multi_%s.pt" % (1, theta_D_key_multi_seed))).numpy())

                                        MI_theta_singles[key].append(theta_single)
                                        MI_theta_multis[key].append(theta_multi)

                                        raw = torch.load(os.path.join(args.out_dir, "%d_theta_D_key_single_%s.pt" % (1, theta_D_key_single_seed)))

                                        num_params[key].append(r["num_params"])
                                        VC[key].append(r["VC"])
                                        sum_weight_norms[key].append(r["sum_weight_norms"])
                                        prod_weight_norms[key].append(r["prod_weight_norms"])

                                        details[key].append(arch_name(arch))
                                    else:
                                        skipped3 += 1

                                    counted += 1

                                model_ind += 1

        print("Counts %s, skipped err %s, thresh %s, gen gap %s" % (counted, skipped1, skipped2, skipped3))

        for metric in metrics:
            locals()["results_%s" % metric] = defaultdict(dict)

        for key in keys:
            print("---")
            print("Key %s" % key)
            GL_curr = np.array(GL[key])
            archs_key = details[key]

            train_losses_curr = np.array(train_losses[key])
            train_accs_curr = np.array(train_accs[key])
            test_losses_curr = np.array(test_losses[key])
            test_accs_curr = np.array(test_accs[key])

            MI_mcs_curr_perlayer = (np.array(MI_mcs[key])) # num models, num layers
            MI_cond_mcs_curr_perlayer = (np.array(MI_cond_mcs[key])) # num models, num layers

            MI_theta_singles_curr_perlayer = (np.array(MI_theta_singles[key])) # num models, num layers
            MI_theta_multis_curr_perlayer = (np.array(MI_theta_multis[key]))

            num_params_curr = np.array(num_params[key])
            VC_curr = np.array(VC[key])
            sum_weight_norms_curr = np.array(sum_weight_norms[key])
            prod_weight_norms_curr = np.array(prod_weight_norms[key])

            scales = [[], []]
            for vname in ["MI_mcs_curr_perlayer", "MI_cond_mcs_curr_perlayer"]:
                v = locals()[vname]

                scales[0].append(v.mean(axis=0) / MI_theta_singles_curr_perlayer[:, :-1].mean(axis=0)) # num layers
                scales[1].append(v.mean(axis=0) / MI_theta_multis_curr_perlayer[:, :-1].mean(axis=0))

            print("Scales for %s %s" % (lamb_lr_curr, lamb_lr_i))
            print(scales)

            MI_mcs_curr = MI_mcs_curr_perlayer[:, -1] # last
            MI_cond_mcs_curr = MI_cond_mcs_curr_perlayer[:, -1]

            MI_theta_singles_curr = MI_theta_singles_curr_perlayer[:, -1]
            MI_theta_multis_curr = MI_theta_multis_curr_perlayer[:, -1]

            combined_singles_mc_curr = (scales[0][0] * MI_theta_singles_curr_perlayer[:, :-1] + MI_mcs_curr_perlayer).min(axis=1)
            combined_single_conds_mc_curr = (scales[0][1] * MI_theta_singles_curr_perlayer[:, :-1] + MI_cond_mcs_curr_perlayer).min(axis=1)

            combined_multis_mc_curr = (scales[1][0] * MI_theta_multis_curr_perlayer[:, :-1] + MI_mcs_curr_perlayer).min(axis=1)
            combined_multi_conds_mc_curr = (scales[1][1] * MI_theta_multis_curr_perlayer[:, :-1] + MI_cond_mcs_curr_perlayer).min(axis=1)

            # noscale
            combined_singles_mc_noscale_curr = (MI_theta_singles_curr_perlayer[:, :-1] + MI_mcs_curr_perlayer).min(axis=1)
            combined_single_conds_mc_noscale_curr = (MI_theta_singles_curr_perlayer[:, :-1] + MI_cond_mcs_curr_perlayer).min(axis=1)

            combined_multis_mc_noscale_curr = (MI_theta_multis_curr_perlayer[:, :-1] + MI_mcs_curr_perlayer).min(axis=1)
            combined_multi_conds_mc_noscale_curr = (MI_theta_multis_curr_perlayer[:, :-1] + MI_cond_mcs_curr_perlayer).min(axis=1)

            if plot:
                fig, axarr = plt.subplots(len(vnames), 2, figsize=(2 * 3, len(vnames) * 3)) # all

                focus_plots = [plt.subplots(1, figsize=(5, 5)) for _ in focus_vnames]

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

                    axarr[v_i, 0].set_title("tau %.3f, r %.3f, p %.3f" % (tau_GL_curr, r_GL_curr, p_GL_curr))

                    # individuals
                    if vname in focus_vnames:
                        vname_i = focus_vnames.index(vname)

                        df1 = create_dataframe(GL_curr, v, archs_key)
                        sns.scatterplot(data=df1, x="GL_curr", y="v", hue="arch", palette="deep",
                                        s=60, ax=focus_plots[vname_i][1])

                        z2 = np.polyfit(GL_curr, v, poly)
                        x = np.linspace(GL_curr.min(), GL_curr.max(), 30)
                        res = np.polyval(z2, x)
                        assert res.shape == x.shape

                        sns.lineplot(x=x, y=res, palette="deep", ax=focus_plots[vname_i][1], linestyle="--")

                        focus_plots[vname_i][1].legend(title="Width", loc="lower right")

                        focus_plots[vname_i][1].set_ylabel(vname)
                        focus_plots[vname_i][1].set_xlabel("GL")
                        focus_plots[vname_i][1].set_title("tau %.3f, r %.3f, p %.3f" % (tau_GL_curr, r_GL_curr, p_GL_curr))

            if plot:
                plt.tight_layout()
                fig.savefig(os.path.join(args.out_dir, "summary_lr_%s_key_%s.pdf" % (lamb_lr_curr, key)), bbox_inches="tight")
                for f_i, (f, _) in enumerate(focus_plots):
                    f.savefig(os.path.join(args.out_dir, "summary_lr_%s_key_%s_%s.pdf" % (lamb_lr_curr, key, focus_vnames[f_i])), bbox_inches="tight")

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

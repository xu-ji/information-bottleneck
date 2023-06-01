import argparse
from toy.util.data import basic_data_instance_MNIST
from toy.util.general import set_seed, device, evaluate, clean, clean_rev, get_weight_norms, compute_factors
from toy.util.swag import *
from toy.util.model import StochasticConvMLP
import torch
import os
from datetime import datetime
from dnn.swag_repo.MI.util import to_bits, create_dataframe
import seaborn as sns

import sys
import numpy as np

from collections import defaultdict
import scipy.stats as stats
import traceback

if False:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

args_form = argparse.ArgumentParser(allow_abbrev=False)

args_form.add_argument("--data", type=str, choices=["digits", "fashion"])

args_form.add_argument("--inds_start", type=int, default=0)
args_form.add_argument("--inds_end", type=int, default=0)

args_form.add_argument("--lr", type=float, default=1e-2)
args_form.add_argument("--MI_const", type=float, default=3.5) #1.5, 1.9 # 2.5
args_form.add_argument("--lamb_init", type=float, default=0.)

args_form.add_argument("--epochs", type=int, default=40)
args_form.add_argument("--swa_start", type=int, default=30)
args_form.add_argument("--out_dir", type=str, default="/network/scratch/x/xu.ji/mutual_info/v22")

args_form.add_argument("--compute_MI_theta_D", default=False, action="store_true")
args_form.add_argument("--compute_MI_theta_D_only_start", type=int, default=0)
args_form.add_argument("--compute_MI_theta_D_only_end", type=int, default=0)

args_form.add_argument("--results", default=False, action="store_true")
args_form.add_argument("--data_only", default=False, action="store_true")

args_form.add_argument("--data_size", type=int, default=8000)


args = args_form.parse_args()
print(args)

print(args.out_dir)


data_instances = list(range(3))

archs = [
[1, 16, 128, 128, 10],
[1, 64, 512, 512, 10],
[1, 32, 256, 256, 10],
]

decays = [0.0, 1e-3]

lamb_lrs = [0, 1e-3]

batch_szs = [128, 32]

seeds = [0, 1]


print("Num models:")
num_models = len(data_instances) * len(seeds) * len(lamb_lrs) * len(archs) * len(decays) * len(batch_szs)
print(num_models)

plt.rc('axes', titlesize=15)
plt.rc('axes', labelsize=15)
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.rc('legend', fontsize=15)
plt.rc('font', size=15)


vnames_pretty = {

    # repr
    "MI_mcs": r"$\hat{I}(X; Z_l^s)$",
    "MI_cond_mcs": r"$\hat{I}(X; Z_l^s | Y)$",

    "MI_jensens": r"$\breve{I}(X; Z_l^s)$",
    "MI_cond_jensens": r"$\breve{I}(X; Z_l^s | Y)$",

    # model
    "MI_theta_singles": r"$\breve{I}(\mathbf{S}; \theta_l^\mathbf{S})$", # jensen upper bound
    "MI_theta_multis": r"$\bar{I}(\mathbf{S}; \theta_l^\mathbf{S})$", # double jensen bound

    "MI_theta_singles_last": r"$\breve{I}(\mathbf{S}; \theta_{D+1}^\mathbf{S})$",
    "MI_theta_multis_last": r"$\bar{I}(\mathbf{S}; \theta_{D+1}^\mathbf{S})$",

    # combined
    "combined_singles_mc": r"$\tilde{I}(\mathbf{S}; \theta_l^\mathbf{S}) + \hat{I}(X; Z_l^s)$",
    "combined_single_conds_mc": r"$\tilde{I}(\mathbf{S}; \theta_l^\mathbf{S}) + \hat{I}(X; Z_l^s | Y)$",

    "combined_singles_jensen": r"$\tilde{I}(\mathbf{S}; \theta_l^\mathbf{S}) + \breve{I}(X; Z_l^s)$",
    "combined_single_conds_jensen": r"$\tilde{I}(\mathbf{S}; \theta_l^\mathbf{S}) + \breve{I}(X; Z_l^s | Y)$",

    "combined_singles_mc_noscale": r"$\breve{I}(\mathbf{S}; \theta_l^\mathbf{S}) + \hat{I}(X; Z_l^s)$",
    "combined_single_conds_mc_noscale": r"$\breve{I}(\mathbf{S}; \theta_l^\mathbf{S}) + \hat{I}(X; Z_l^s | Y)$",

    "combined_singles_jensen_noscale": r"$\breve{I}(\mathbf{S}; \theta_l^\mathbf{S}) + \breve{I}(X; Z_l^s)$",
    "combined_single_conds_jensen_noscale": r"$\breve{I}(\mathbf{S}; \theta_l^\mathbf{S}) + \breve{I}(X; Z_l^s | Y)$",

}


################################
# Training models
################################


model_ind = 0
for lamb_lr in lamb_lrs:
    for arch_i, arch in enumerate(archs):
        for decay in decays:
            for batch_sz in batch_szs:

                for data_instance in data_instances:
                    for seed in seeds:

                        if model_ind in range(args.inds_start, args.inds_end): # exclusive

                            savepath = os.path.join(args.out_dir, "results_%d.pt" % model_ind)
                            if os.path.exists(savepath) and (arch_i != 0): # don't ever skip if need to update arch, todo
                                print("Skipping %d" % model_ind)
                                model_ind += 1
                                continue

                            train_dl, test_dl = basic_data_instance_MNIST(args, batch_sz, data_instance, args.data_size)

                            if args.data_only:
                                exit(0)

                            print(("num batches", len(train_dl), len(test_dl)))

                            print("Doing model_ind %d, %s" % (model_ind, datetime.now()))
                            print((seed, lamb_lr, arch, decay, batch_sz))
                            sys.stdout.flush()

                            assert arch[-1] == args.C
                            set_seed(seed)

                            model = StochasticConvMLP(arch, args.in_dim).to(device).train()
                            opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=decay)

                            try:
                                model, swag_model, diagnostics = train_model_swag(model, arch, opt, train_dl, test_dl, args, lamb_lr, verbose=True)
                            except Exception as e:
                                print(e)
                                print("Error, skipping")
                                model_ind += 1
                                continue

                            train_acc, train_loss = evaluate(model, train_dl, args, "train", plot=False)
                            test_acc, test_loss = evaluate(model, test_dl, args, "test", plot=False)

                            print("train: %.3f %.3f, test: %.3f %.3f, MI: mc %s %s jensen %s %s, diagnostics: test %.3f %.3f swag %.3f %.3f" %
                                  (train_acc, train_loss, test_acc, test_loss,
                                   diagnostics["MI_mc"], diagnostics["MI_cond_mc"], diagnostics["MI_jensen"], diagnostics["MI_cond_jensen"],
                                   diagnostics["test_losses"][-1], diagnostics["test_accs"][-1],
                                   diagnostics["test_loss_swag"], diagnostics["test_acc_swag"],
                                   ))

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

                                "train_acc": train_acc,
                                "train_loss": train_loss,
                                "test_acc": test_acc,
                                "test_loss": test_loss,

                                "gen_gap_err": train_acc - test_acc, # = 1 - test_acc - (1 - train_acc)
                                "gen_gap_loss": test_loss - train_loss,

                                "diagnostics": diagnostics
                            }

                            results["num_params"] = num_params
                            results["VC"] = VC
                            results["sum_weight_norms"] = sum_weight_norms
                            results["prod_weight_norms"] = prod_weight_norms

                            torch.save(results, savepath)

                        model_ind += 1


################################
# Model compression
################################

if args.compute_MI_theta_D:
    num_samples = 5
    set_seed(1)

    model_ind = 0
    setting_i = 0
    for lamb_lr in lamb_lrs:
        for arch in archs:
            for decay in decays:
                for batch_sz in batch_szs:

                    doing_setting = setting_i in list(range(args.compute_MI_theta_D_only_start, args.compute_MI_theta_D_only_end))

                    if doing_setting: swag_models = defaultdict(list)
                    for data_instance in data_instances:
                        for seed in seeds:
                            if doing_setting:
                                results_f = os.path.join(args.out_dir, "results_%d.pt" % model_ind)
                                print("loading %s" % results_f)
                                results = torch.load(results_f)
                                swag_models[seed].append(results["swag_model"])

                                assert results["arch"] != [1, 128, 1028, 1028, 10]

                            model_ind += 1 # increment even if not doing

                    if doing_setting:
                        print("%s" % datetime.now())
                        sys.stdout.flush()

                        for seed in seeds:
                            theta_D_key_single_seed = clean("%s_%s_%s_%s" % (seed, lamb_lr, arch, decay))

                            MI_theta_D_single_seed = compute_MI_theta_D_single_seed_jensen(swag_models[seed], num_samples)
                            torch.save(MI_theta_D_single_seed, os.path.join(args.out_dir,"%d_theta_D_key_single_%s.pt" % (1, theta_D_key_single_seed)))

                        theta_D_key_multi_seed = clean("%s_%s_%s" % (lamb_lr, arch, decay))
                        MI_theta_D_multi_seed = compute_MI_theta_D_multiseed_jensen(swag_models, num_samples)
                        torch.save(MI_theta_D_multi_seed, os.path.join(args.out_dir, "%d_theta_D_key_multi_%s.pt" % (1, theta_D_key_multi_seed)))

                        print(("MI results for %s (%s): single %s, multi %s" % (setting_i, theta_D_key_multi_seed, MI_theta_D_single_seed, MI_theta_D_multi_seed)))

                    setting_i += 1


################################
# Results
################################


if args.results:
    sns.set_style("dark")

    plot = True
    print_summary = False

    vnames_base_invariant = [
        "MI_theta_singles_last",
        "MI_theta_multis_last",
    ]

    vnames_base_extra = [
        "num_params",
        "VC",
        "sum_weight_norms",
        "prod_weight_norms"
    ]

    vnames_base = ["MI_mcs", "MI_cond_mcs", "MI_jensens", "MI_cond_jensens",
        "MI_theta_singles", "MI_theta_multis"] + vnames_base_invariant + vnames_base_extra

    vnames = vnames_base + [
        "combined_singles_mc", "combined_single_conds_mc",
        "combined_singles_jensen", "combined_single_conds_jensen",

        "combined_singles_mc_noscale", "combined_single_conds_mc_noscale",
        "combined_singles_jensen_noscale", "combined_single_conds_jensen_noscale",

        # others
        "combined_multis_mc", "combined_multi_conds_mc",
        "combined_multis_jensen", "combined_multi_conds_jensen",

        "combined_multis_mc_noscale", "combined_multi_conds_mc_noscale",
        "combined_multis_jensen_noscale", "combined_multi_conds_jensen_noscale",
    ]

    thresh = 0.0
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
            model_ind = 0

            metrics = ["tau_GL", "r_GL", "p_GL"]

            keys = {}
            for lamb_lr in lamb_lrs:
                for arch_i, arch in enumerate(archs):
                    for decay_i, decay in enumerate(decays):
                        for batch_i, batch_sz in enumerate(batch_szs):

                            for data_i, data_instance in enumerate(data_instances):
                                for seed_i, seed in enumerate(seeds):

                                    if lamb_lr == lamb_lr_curr:

                                        r = torch.load(os.path.join(args.out_dir, "results_%d.pt" % model_ind))

                                        theta_D_key_single_seed = clean("%s_%s_%s_%s" % (seed, lamb_lr, arch, decay))
                                        theta_single = to_bits(torch.load(
                                            os.path.join(args.out_dir,
                                                         "%d_theta_D_key_single_%s.pt" % (
                                                         1,
                                                         theta_D_key_single_seed))))

                                        theta_D_key_multi_seed = clean("%s_%s_%s" % (lamb_lr, arch, decay))
                                        theta_multi = to_bits(torch.load(
                                            os.path.join(args.out_dir,
                                                         "%d_theta_D_key_multi_%s.pt" % (
                                                         1, theta_D_key_multi_seed))))

                                        if (r["train_acc"] > thresh):

                                            key = get_key_MNIST(arch_i, decay_i, batch_i, data_i, seed_i, to_vary)
                                            keys[key] = 1

                                            GL[key].append(r["gen_gap_loss"])

                                            train_losses[key].append(r["train_loss"])
                                            train_accs[key].append(r["train_acc"])
                                            test_losses[key].append(r["test_loss"])
                                            test_accs[key].append(r["test_acc"])

                                            MI_mcs[key].append(to_bits(r["diagnostics"]["MI_mc"]))

                                            MI_cond_mcs[key].append(to_bits(r["diagnostics"]["MI_cond_mc"]))
                                            MI_jensens[key].append(to_bits(r["diagnostics"]["MI_jensen"]))
                                            MI_cond_jensens[key].append(to_bits(r["diagnostics"]["MI_cond_jensen"]))


                                            MI_theta_singles[key].append(theta_single[0].item())

                                            MI_theta_multis[key].append(theta_multi[0].item())

                                            MI_theta_singles_last[key].append(theta_single[1].item())
                                            MI_theta_multis_last[key].append(theta_multi[1].item())

                                            details[key].append(arch_name_MNIST(arch))

                                            num_params[key].append(r["num_params"])
                                            VC[key].append(r["VC"])
                                            sum_weight_norms[key].append(r["sum_weight_norms"])
                                            prod_weight_norms[key].append(r["prod_weight_norms"])

                                        else:
                                            skipped2 += 1

                                        counted += 1

                                    model_ind += 1

            print("Counts %s, skipped err %s, thresh %s" % (counted, skipped1, skipped2))

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

                MI_theta_singles_curr = (np.array(MI_theta_singles[key])) # each one combined with the above 4
                MI_theta_multis_curr = (np.array(MI_theta_multis[key]))

                MI_theta_singles_last_curr = (np.array(MI_theta_singles_last[key]))  # each one combined with the above 4
                MI_theta_multis_last_curr = (np.array(MI_theta_multis_last[key]))

                #print("MI_thetas info for %s" % key)
                #print((MI_theta_singles_curr,MI_theta_multis_curr, MI_theta_singles_last_curr, MI_theta_multis_last_curr))

                scales = [[], []]
                for vname in ["MI_mcs_curr", "MI_cond_mcs_curr", "MI_jensens_curr", "MI_cond_jensens_curr"]:
                    v = locals()[vname]

                    scales[0].append(v.mean() / MI_theta_singles_curr[np.isfinite(MI_theta_singles_curr)].mean())
                    scales[1].append(v.mean() / MI_theta_multis_curr[np.isfinite(MI_theta_multis_curr)].mean())

                #print("Scales for %s %s" % (lamb_lr_curr, lamb_lr_i))
                #print(scales)

                combined_singles_mc_curr = scales[0][0] * MI_theta_singles_curr + MI_mcs_curr
                combined_single_conds_mc_curr = scales[0][1] * MI_theta_singles_curr + MI_cond_mcs_curr

                combined_singles_jensen_curr = scales[0][2] * MI_theta_singles_curr + MI_jensens_curr
                combined_single_conds_jensen_curr = scales[0][3] * MI_theta_singles_curr + MI_cond_jensens_curr

                combined_multis_mc_curr = scales[1][0] * MI_theta_multis_curr + MI_mcs_curr
                combined_multi_conds_mc_curr = scales[1][1] * MI_theta_multis_curr + MI_cond_mcs_curr

                combined_multis_jensen_curr = scales[1][2] * MI_theta_multis_curr + MI_jensens_curr
                combined_multi_conds_jensen_curr = scales[1][3] * MI_theta_multis_curr + MI_cond_jensens_curr

                for seed_chosen in range(combined_single_conds_mc_curr.shape[0]):
                    if abs(combined_single_conds_mc_curr[seed_chosen] - combined_single_conds_mc_curr.mean()) / combined_single_conds_mc_curr.std() >= 2:
                        print(("outlier", combined_single_conds_mc_curr[seed_chosen], details[key][seed_chosen]))

                # noscale

                combined_singles_jensen_noscale_curr = MI_theta_singles_curr + MI_jensens_curr
                combined_single_conds_jensen_noscale_curr = MI_theta_singles_curr + MI_cond_jensens_curr

                combined_multis_jensen_noscale_curr = MI_theta_multis_curr + MI_jensens_curr
                combined_multi_conds_jensen_noscale_curr = MI_theta_multis_curr + MI_cond_jensens_curr

                combined_singles_mc_noscale_curr = MI_theta_singles_curr + MI_mcs_curr
                combined_single_conds_mc_noscale_curr = MI_theta_singles_curr + MI_cond_mcs_curr

                combined_multis_mc_noscale_curr = MI_theta_multis_curr + MI_mcs_curr
                combined_multi_conds_mc_noscale_curr = MI_theta_multis_curr + MI_cond_mcs_curr

                num_params_curr = np.array(num_params[key])
                VC_curr = np.array(VC[key])
                sum_weight_norms_curr = np.array(sum_weight_norms[key])
                prod_weight_norms_curr = np.array(prod_weight_norms[key])

                if plot:
                    fig, axarr = plt.subplots(len(vnames), 2, figsize=(2 * 3, len(vnames) * 3)) # all
                    fig2, axarr2 = plt.subplots(1, figsize=(5, 5)) # MI_mcs
                    fig3, axarr3 = plt.subplots(1, figsize=(5, 5)) # combined_single_conds_mc
                    fig4, axarr4 = plt.subplots(1, figsize=(5, 5)) # combined_single_conds_mc

                for v_i, vname in enumerate(vnames):
                    v = locals()["%s_curr" % vname]

                    for metric in metrics: # wht
                        locals()["%s_curr" % metric] = np.nan

                    try:
                        sub = np.isfinite(v)

                        #print("%s num finite: %s" % (vname, sub.sum().item()))
                        #if vname in ["MI_theta_singles", "MI_theta_multis"]:
                            #print("Values:")
                            #print(v)
                            #print("%.10f %.10f" % (v[sub].max().item(), v[sub].min().item()))

                        tau_GL_curr, _ = stats.kendalltau(GL_curr[sub], v[sub])
                        r_GL_curr, _ = stats.spearmanr(GL_curr[sub], v[sub])
                        p_GL_curr, _ = stats.pearsonr(GL_curr[sub], v[sub])

                    except Exception as e:
                        print("Printing error")
                        print(traceback.format_exc())
                        #tau_GL_curr = np.nan
                        #r_GL_curr = np.nan
                        #p_GL_curr = np.nan

                        continue

                    for metric in metrics:
                        locals()["results_%s" % metric][vname][key] = locals()["%s_curr" % metric]

                    if not print_summary:
                        pretty_name = vname
                        if vname in vnames_pretty: pretty_name = vnames_pretty[vname]
                        else: pretty_name = clean_rev(vname)
                        print("%s \t& %.4f & %.4f & %.4f \\\\" % (pretty_name, r_GL_curr, p_GL_curr, tau_GL_curr))
                        print("")

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
                        if vname == "MI_mcs":
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
                                axarr2.legend().set_visible(False)

                            axarr2.set_ylabel(vnames_pretty[vname])
                            axarr2.set_xlabel("Generalization gap in loss")
                            axarr2.set_title("Pearson correlation: %.3f" % (p_GL_curr))
                            #axarr2.set_xticklabels(axarr2.get_xticklabels(), rotation=45)

                        if vname == "combined_single_conds_mc":
                            #axarr3.scatter(GL_curr, v)
                            df2 = create_dataframe(GL_curr, v, archs_key)
                            sns.scatterplot(data=df2, x="GL_curr", y="v",
                                            hue="arch", palette="deep", s=60, ax=axarr3)

                            z3 = np.polyfit(GL_curr, v, poly)
                            x = np.linspace(GL_curr.min(), GL_curr.max(), 30)
                            res = np.polyval(z3, x)
                            assert res.shape == x.shape

                            #axarr3.plot(x, res, "r--")
                            sns.lineplot(x=x, y=res, palette="deep", ax=axarr3, linestyle="--")
                            #axarr3.legend(title="Width", loc={0: "center right", 1:"center right", 2:"center right"}[lamb_lr_i])

                            if lamb_lr_i == 0:
                                axarr3.legend(title="Width", loc="lower right")
                            else:
                                axarr3.legend().set_visible(False)

                            axarr3.set_ylabel(vnames_pretty[vname])
                            axarr3.set_xlabel("Generalization gap in loss")
                            axarr3.set_title("Pearson correlation: %.3f" % (p_GL_curr))

                        if vname == "combined_single_conds_jensen":
                            #axarr3.scatter(GL_curr, v)
                            df2 = create_dataframe(GL_curr, v, archs_key)
                            sns.scatterplot(data=df2, x="GL_curr", y="v",
                                            hue="arch", palette="deep", s=60, ax=axarr4)

                            z3 = np.polyfit(GL_curr, v, poly)
                            x = np.linspace(GL_curr.min(), GL_curr.max(), 30)
                            res = np.polyval(z3, x)
                            assert res.shape == x.shape

                            #axarr3.plot(x, res, "r--")
                            sns.lineplot(x=x, y=res, palette="deep", ax=axarr4, linestyle="--")
                            #axarr3.legend(title="Width", loc={0: "center right", 1:"center right", 2:"center right"}[lamb_lr_i])

                            if lamb_lr_i == 0:
                                axarr4.legend(title="Width", loc="lower right")
                            else:
                                axarr4.legend().set_visible(False)

                            axarr4.set_ylabel(vnames_pretty[vname])
                            axarr4.set_xlabel("Generalization gap in loss")
                            axarr4.set_title("Pearson correlation: %.3f" % (p_GL_curr))


                if plot:
                    plt.tight_layout()
                    fig.savefig(os.path.join(args.out_dir, "summary_lr_%s_key_%s.pdf" % (lamb_lr_curr, key)), bbox_inches="tight")
                    fig2.savefig(os.path.join(args.out_dir, "summary_lr_%s_key_%s_MI_mcs.pdf" % (lamb_lr_curr, key)), bbox_inches="tight")
                    fig3.savefig(os.path.join(args.out_dir, "summary_lr_%s_key_%s_combined_single_conds_mc.pdf" % (lamb_lr_curr, key)), bbox_inches="tight")
                    fig4.savefig(os.path.join(args.out_dir, "summary_lr_%s_key_%s_combined_single_conds_jensen.pdf" % (lamb_lr_curr, key)), bbox_inches="tight")

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

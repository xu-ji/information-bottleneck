import torch, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_style("darkgrid")

model_ind = 131
out_dir = "/network/scratch/x/xu.ji/mutual_info/v6c"

r = torch.load(os.path.join(out_dir, "results_%d.pt" % model_ind))

fig, axarr = plt.subplots(1, figsize=(4, 1 * 4))
l_MIs = r["diagnostics"]["l_MIs"]
#axarr[0].plot(r["diagnostics"]["l_MIs"])
sns.lineplot(x=range(len(l_MIs)), y=l_MIs, palette="deep", ax=axarr)
axarr.set_xlabel("Training steps")
axarr.set_ylabel("$\hat{I}(X; Z_l^s)$")
fig.savefig(os.path.join(out_dir, "pretty_0_training_%d.pdf" % model_ind), bbox_inches="tight")

fig2, axarr2 = plt.subplots(1, figsize=(4, 1 * 4))
lambs = r["diagnostics"]["lambs"]
#axarr[1].plot(r["diagnostics"]["lambs"])
sns.lineplot(x=range(len(lambs)), y=lambs, palette="deep", ax=axarr2)
axarr2.set_xlabel("Training steps")
axarr2.set_ylabel("$\lambda$")
fig2.savefig(os.path.join(out_dir, "pretty_1_training_%d.pdf" % model_ind), bbox_inches="tight")

#for plot_i, (plot_name, plot_values) in enumerate(diagnostics.items()):
#    axarr[plot_i].plot(plot_values)
#    axarr[plot_i].set_ylabel(plot_name)


plt.tight_layout()
plt.close("all")
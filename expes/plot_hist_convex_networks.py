import torch
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

save_path = "./results/depths_widths/"
res_convex = torch.load(save_path + "results_convex_5000.pt")
res_icnn = torch.load(save_path + "results_icnn_5000.pt")
times = torch.load(save_path + "times_computations_5000.pt")
res_convex = res_convex[:6, :]
res_icnn = res_icnn[:6, :]
res_convex_proba = res_convex/10000.
res_icnn_proba = res_icnn/10000.


params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}
pylab.rcParams.update(params)
# fig, axes = plt.subplots(1, 3, figsize=(9, 6), gridspec_kw={
#                          'width_ratios': [1., 0.05, 1]})

fig = plt.figure(figsize=(5, 8))
# xlabs = [2, 3, 4, 5, 6, 7]
ylabs = [2, 3, 4, 5, 6, 7, 8, 9, 10]
xlabs = [2, 3, 4, 5, 6, 7]

# Plot the first heatmap
# sns.heatmap(res_convex_proba.T, ax=axes[0], annot=res_convex.T.type(
#     torch.int), fmt="", vmin=0, vmax=0.1, cmap="rocket_r", cbar=False)
ax = sns.heatmap(res_convex_proba.T,  annot=res_convex.T.type(
    torch.int), fmt="", vmin=0, vmax=0.1, cmap="rocket_r", cbar=False)
# axes[0].set_title("")
# axes[0].set_xticklabels(xlabs)
# axes[0].set_yticklabels(ylabs)
# axes[0].set_xlabel(f"width")
# axes[0].set_ylabel(f"depth", rotation=0)
# axes[0].set_title(f"Convex $\mathrm{{ReLU}}$ networks")
ax.set_xticklabels(xlabs)
ax.set_yticklabels(ylabs)
ax.set_xlabel(f"Width", fontsize=12)
ax.set_ylabel(f"Depth", rotation=0, fontsize=12)
# plt.title(f"Convex $\mathrm{{ReLU}}$ networks", fontsize=12)
# Plot the second heatmap
# sns.heatmap(res_icnn_proba.T, ax=axes[2], annot=res_icnn.T.type(
#     torch.int),  fmt="",  vmin=0, vmax=0.1, cmap="rocket_r", cbar=False)
# # axes[1].set_title("Heatmap 2")
# axes[2].set_xticklabels(xlabs)
# axes[2].set_yticklabels(ylabs)
# axes[2].set_xlabel(f"width", fontsize=10)
# axes[2].set_ylabel(f"depth", rotation=0, fontsize=10)
# axes[2].set_title("ICNNs")

# Add a single colorbar for both heatmaps
# cbar = fig.colorbar(axes[0].collections[0],
#                     cax=axes[1], orientation='vertical')


plt.tight_layout()
plt.show(block=False)
fig.savefig(save_path + "exp_convex_heat.pdf")


# params = {'legend.fontsize': 'x-large',
#           'figure.figsize': (15, 5),
#           'axes.labelsize': 'x-large',
#           'axes.titlesize': 'x-large',
#           'xtick.labelsize': 'x-large',
#           'ytick.labelsize': 'x-large'}
# pylab.rcParams.update(params)
# fig, axes = plt.subplots(1, 1, figsize=(14, 6))

# xlabs = [2, 3, 4, 5, 6, 7]
# ylabs = [2, 3, 4, 5, 6, 7]
# # Plot the first heatmap
# sns.heatmap(times.T, ax=axes, annot=times.T.type(
#     torch.float16), fmt="", vmin=0, vmax=0.1, cmap="rocket_r", cbar=False)
# # axes[0].set_title("")
# axes.set_xticklabels(xlabs)
# axes.set_yticklabels(ylabs)
# axes.set_xlabel(f"$n_1$")
# axes.set_ylabel(f"$n_2$", rotation=0)
# axes.set_title(f"Avg time")

# plt.tight_layout()
# plt.show(block=False)
# fig.savefig(save_path + "exp_times_heat.pdf")

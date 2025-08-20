import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tqdm import trange
from time import time
import matplotlib.pyplot as plt

from beyondicnn.edges_subdivision import skeletal_subdivision, get_e_sv
from beyondicnn.check_convexity import check_convexity
from beyondicnn.utils_viz import *
from beyondicnn.models import FeedForwardNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

results_save_path = f"./results/"

os.makedirs(results_save_path, exist_ok=True)

PLOTS = False
n_test = 100
times = {}

torch.cuda.manual_seed(0)

input_dim = 2

bbox = -100, 100

widths_lay1 = [2**k for k in range(10)]

times_computations = torch.zeros(len(widths_lay1))
res_convex = torch.zeros(len(widths_lay1))
res_icnn = torch.zeros(len(widths_lay1))

for i, w1 in enumerate(widths_lay1):
    w2 = w1
    convex_nets = torch.zeros(n_test)
    icnn_nets = torch.zeros(n_test)
    sanity_check = torch.zeros(n_test)
    t1 = time()
    for k in trange(n_test):
        f = FeedForwardNet(input_dim=input_dim, widths=[w1, w2]).to(device)

        vs, edges, v_sv = skeletal_subdivision(
            f, device=device,  bbox=bbox, plot=True, return_intermediate=False)
        e_sv = get_e_sv(v_sv, edges)

        convexity, res = check_convexity(f, e_sv)

        convex_nets[k] = convexity
        if torch.all(f.fc3.weight.data >= 0):
            if torch.all(f.fc2.weight.data >= 0):
                sanity_check[k] = 1.0

        if convexity:

            if torch.all(f.fc3.weight.data >= 0):
                if torch.all(f.fc2.weight.data >= 0):
                    icnn_nets[k] = 1.0
                else:
                    if PLOTS:
                        if not torch.all((f.fc3.weight.data.T *
                                          f.fc2.weight.data).sum(axis=0) >= 0):
                            if not torch.all(f.fc2.weight.data <= 0):
                                print(f.fc2.weight,
                                      f.fc2.bias, f.fc3.weight)
                                print("heho", (f.fc3.weight.data.T *
                                               f.fc2.weight.data).sum(axis=0) >= 0)

                                intermediates = skeletal_subdivision(
                                    f, device='cpu',  bbox=bbox, plot=True, return_intermediate=True)
                                for l in range(1, len(intermediates)+1):
                                    vs = intermediates[l][0]
                                    edges = intermediates[l][1]
                                    v_sv = intermediates[l][2]
                                    plot_verts_and_edges(
                                        vs, edges, v_labels=v_sv[:, 2*input_dim:])
                                fig, ax = plt.subplots()
                                plot_field(f, ax, dim=2, model_name="Trained net",
                                           xmin=-6, xmax=6)
                                # fig.suptitle(r"Field given by $f_{{EX}}$")
                                ax.set_xticks(np.linspace(-6, 6, 20))
                                ax.set_yticks(np.linspace(-6, 6, 20))
                                ax.set_xlabel(r"$x_1$")
                                ax.set_ylabel(r"$x_2$")
                                plt.show()
                                fig, ax = plt.subplots(
                                    subplot_kw={"projection": "3d"})
                                plot_field(f, ax, dim=3, model_name="Trained net",
                                           print_steps=30, xmin=-6, xmax=6)
                                ax.set_xlabel(r"$x_1$")
                                ax.set_ylabel(r"$x_2$")
                                fig.suptitle(r"$f_{{EX}}$")
                                plt.show(block=True)

    print(w1)
    times_computations[i] = (time()-t1) / n_test
    print(convex_nets.sum())
    print(icnn_nets.sum())
    print(sanity_check.sum())
    res_convex[i] = (convex_nets.sum())
    res_icnn[i] = (icnn_nets.sum())
    torch.save(res_convex, results_save_path +
               f"results_convex_2_{100}.pt")
    torch.save(res_icnn, results_save_path + f"results_icnn_2_{100}.pt")
    torch.save(times_computations, results_save_path +
               f"times_computations_{100}.pt")

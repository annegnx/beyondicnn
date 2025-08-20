import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tqdm import trange, tqdm
from time import time

from beyondicnn.edges_subdivision import skeletal_subdivision, get_e_sv
from beyondicnn.check_convexity import check_convexity
from beyondicnn.utils_viz import *
from beyondicnn.models import FeedForwardNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

results_save_path = f"./results/depths_widths/"

os.makedirs(results_save_path, exist_ok=True)


PLOTS = False
n_test = 5000
times = {}

torch.cuda.manual_seed(0)

bbox = -100, 100

depths = [2, 3, 4, 5, 6, 7, 8, 9, 10]
widths_lay = [2, 3, 4, 5, 6, 7, 8, 9, 10]
input_dim = 3
times_computations = torch.zeros((len(widths_lay), len(depths)))
res_convex = torch.zeros((len(widths_lay), len(depths)))
res_icnn = torch.zeros((len(widths_lay), len(depths)))

for i, w1 in enumerate(widths_lay):
    for j, depth in enumerate(depths):
        print("deoth", depth, "width", w1)
        convex_nets = torch.zeros(n_test)
        icnn_nets = torch.zeros(n_test)
        sanity_check = torch.zeros(n_test)
        t1 = time()
        for k in trange(n_test):
            f = FeedForwardNet(input_dim=input_dim, widths=[
                w1 for k in range(depth)]).to(device)

            vs, edges, v_sv = skeletal_subdivision(
                f, device=device,  bbox=bbox, plot=True, return_intermediate=False)
            e_sv = get_e_sv(v_sv, edges)

            convexity, res = check_convexity(f, e_sv)

            convex_nets[k] = convexity
            sanity_check[k] = True
            for layer in f.layers[1::]:
                if torch.all(layer.weight.data >= 0):
                    sanity_check[k] = True
                else:
                    sanity_check[k] = False
                    break

            if convexity:
                is_icnn = True
                for layer in f.layers[1::]:
                    if torch.all(layer.weight.data >= 0):
                        pass
                    else:
                        is_icnn = False
                        break

                icnn_nets[k] = int(is_icnn)

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

        times_computations[i, j] = (time()-t1) / n_test
        print(convex_nets.sum())
        print(icnn_nets.sum())
        print(sanity_check.sum())
        res_convex[i, j] = (convex_nets.sum())
        res_icnn[i, j] = (icnn_nets.sum())
        torch.save(res_convex, results_save_path +
                   f"results_convex_{n_test}.pt")
        torch.save(res_icnn, results_save_path + f"results_icnn_{n_test}.pt")
        torch.save(times_computations, results_save_path +
                   f"times_computations_{n_test}.pt")

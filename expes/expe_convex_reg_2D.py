import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tqdm import trange, tqdm
from time import time

from beyondicnn.edges_subdivision import skeletal_subdivision, get_e_sv
from beyondicnn.check_convexity import get_convexity_constraints, check_convexity
from beyondicnn.convex_reg import convex_regularisation
from beyondicnn.models import FeedForwardNet
from beyondicnn.utils_viz import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

results_save_path = f"./results/reg_cpwl/"

os.makedirs(results_save_path, exist_ok=True)

torch.random.manual_seed(0)


input_dim = 2
bbox = -10, 10
convexity = False

# Generate a convex CPWL function (at random)
while not convexity:
    f = FeedForwardNet(input_dim=input_dim, widths=[4, 8]).to(device)
    f.init_weights()
    vs, edges, v_sv = skeletal_subdivision(
        f, device=device,  bbox=bbox, plot=True, return_intermediate=False)
    e_sv = get_e_sv(v_sv, edges)
    convexity, res = check_convexity(f, e_sv)

for layer in f.fcs:
    print(layer.weight.data)

f.eval()
target_function = f

fig, ax = plt.subplots()
plot_field(f, ax, dim=2, model_name="Trained net",
           xmin=-6, xmax=6)
ax.set_xticks(np.linspace(-6, 6, 20))
ax.set_yticks(np.linspace(-6, 6, 20))
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
plt.show()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
plot_field(f, ax, dim=3, model_name="Trained net",
           print_steps=30, xmin=-6, xmax=6)
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
fig.suptitle(r"$f_{{EX}}$")
plt.show(block=True)


input_dim = 2
widths = [4, 8]


num_of_tests = 1
n_train = 200000
n_val = 20000
training_set = 20 * torch.rand((n_train, 2)) - 10
val_set = 20 * torch.rand((n_val, 2)) - 10

target_val = target_function(val_set)

batch_size = 256
epochs = 25

loss_train = torch.zeros((num_of_tests, epochs))
loss_val = torch.zeros((num_of_tests, epochs))
loss_reg = torch.zeros((num_of_tests, epochs))
loss_train_icnn = torch.zeros((num_of_tests, epochs))
loss_val_icnn = torch.zeros((num_of_tests, epochs))
loss_train_unconstrained = torch.zeros((num_of_tests, epochs))
loss_val_unconstrained = torch.zeros((num_of_tests, epochs))

for n in range(num_of_tests):
    loss = torch.nn.L1Loss()
    loss_icnn = torch.nn.L1Loss()
    loss_unconstrained = torch.nn.L1Loss()
    lmbda = 0.1

    net = FeedForwardNet(input_dim=input_dim, widths=widths)
    torch.save(net.state_dict(), "net.pth")
    icnn_net = FeedForwardNet(input_dim=input_dim, widths=widths)
    icnn_net.load_state_dict(torch.load("net.pth"))  # share same init
    unconstrained_net = FeedForwardNet(input_dim=input_dim, widths=widths)
    unconstrained_net.load_state_dict(torch.load("net.pth"))  # share same init

    opt = torch.optim.Adam(net.parameters(), lr=1e-3, betas=(0.7, 0.9))
    opt_icnn = torch.optim.Adam(icnn_net.parameters(), lr=1e-3, betas=(0.7, 0.9))
    opt_unconstrained = torch.optim.Adam(
        unconstrained_net.parameters(), lr=1e-3, betas=(0.7, 0.9))

    for i in trange(epochs):

        loss_avg = 0
        reg_avg = 0.
        loss_avg_icnn = 0
        loss_avg_unconstrianed = 0.
        for n_batch in range(n_train // batch_size):

            batch = training_set[n_batch * batch_size:: (n_batch+1) * batch_size, :]
            batch_icnn = batch.clone()
            batch_icnn.requires_grad_(True)
            batch.requires_grad_(True)
            batch_unconstrained = batch.clone()
            batch_unconstrained.requires_grad_(True)

            target = target_function(batch)
            target_icnn = target_function(batch_icnn)
            target_unconstrained = target_function(batch_unconstrained)

            output = net(batch)

            loss_ = loss(output, target)
            reg, constraints = convex_regularisation(net, device=device)
            total_loss = loss_ + lmbda * reg

            opt.zero_grad()
            total_loss.backward()
            opt.step()

            output_icnn = icnn_net(batch_icnn)
            loss_icnn_ = loss_icnn(output_icnn, target_icnn)
            opt_icnn.zero_grad()
            loss_icnn_.backward()
            opt_icnn.step()

            output_unconstrained = unconstrained_net(batch_unconstrained)
            loss_unconstrained_ = loss_unconstrained(
                output_unconstrained, target_unconstrained)
            opt_unconstrained.zero_grad()
            loss_unconstrained_.backward()
            opt_unconstrained.step()

            icnn_net.w_clip()
            loss_avg += loss_.item()
            loss_avg_icnn += loss_icnn_.item()
            loss_avg_unconstrianed += loss_unconstrained_.item()
            reg_avg += reg.item()

        if i % 10 == 0:
            print("Reg Net")
            print("loss", total_loss.item())
            print("reg",  reg_avg / (n_train // batch_size))
            print("datafit",  loss_avg / (n_train // batch_size))
            print("ICNN")
            print("loss ICNN",  loss_avg_icnn / (n_train // batch_size))
            print("Unconstrained Net")
            print("loss Unconstrained",  loss_avg_unconstrianed / (n_train // batch_size))
        loss_train[n, i] = loss_avg / (n_train // batch_size)
        loss_train_icnn[n, i] = loss_avg_icnn / (n_train // batch_size)
        loss_reg[n, i] = reg_avg / (n_train // batch_size)
        loss_train_unconstrained[n, i] = loss_avg_unconstrianed / \
            (n_train // batch_size)

        val_set.requires_grad_(True)
        output_val = net(val_set)

        output_icnn_val = icnn_net(val_set)
        output_unconstrained_val = unconstrained_net(val_set)

        loss_val[n, i] = ((output_val - target_val)**2).sum(dim=1).mean().item()
        loss_val_icnn[n, i] = ((output_icnn_val - target_val)
                               ** 2).sum(dim=1).mean().item()
        loss_val_unconstrained[n, i] = (
            (output_unconstrained_val - target_val)**2).sum(dim=1).mean().item()


# fig = plt.figure()
# plt.plot(range(epochs), loss_val.mean(dim=0), color='r', label="Validation loss (ours)")
# plt.plot(range(epochs), loss_val_icnn.mean(dim=0),
#          color='b', label="Validation loss (ICNN)")
# plt.plot(range(epochs), loss_train.mean(dim=0), color='g', label="Train loss (ours)")
# plt.plot(range(epochs), loss_train_icnn.mean(
#     dim=0),  label="Train loss (ICNN)", color="k")
# plt.plot(range(epochs), loss_train_unconstrained.mean(
#     dim=0),  label="Train loss (unconstrained)", color="b")
# plt.plot(range(epochs), loss_val_unconstrained.mean(
#     dim=0),  label="Validation loss (unconstrained)", color="c")
# plt.plot(range(epochs), loss_reg.mean(
#     dim=0),  label="Regularisation (ours)", color="m")
# plt.fill_between(range(epochs), loss_val.mean(dim=0) - loss_val.std(dim=0),
#                  loss_val.mean(dim=0) + loss_val.std(dim=0), color="r", alpha=0.2)
# plt.fill_between(range(epochs), loss_val_icnn.mean(dim=0) - loss_val_icnn.std(dim=0),
#                  loss_val_icnn.mean(dim=0) + loss_val_icnn.std(dim=0), color="b", alpha=0.2)
# plt.fill_between(range(epochs), loss_train.mean(dim=0) - loss_train.std(dim=0),
#                  loss_train.mean(dim=0) + loss_train.std(dim=0), color="g", alpha=0.2)
# plt.fill_between(range(epochs), loss_train_icnn.mean(dim=0) - loss_train_icnn.std(dim=0),
#                  loss_train_icnn.mean(dim=0) + loss_train_icnn.std(dim=0), color="k", alpha=0.2)
# plt.fill_between(range(epochs), loss_reg.mean(dim=0) - loss_reg.std(dim=0),
#                  loss_reg.mean(dim=0) + loss_reg.std(dim=0), color="m", alpha=0.2)
# plt.legend()
# plt.tight_layout(pad=0.)
# fig.savefig(results_save_path + "loss.pdf")
# plt.show(block=False)

x, y = torch.linspace(-10, 10, 1000), torch.linspace(-10, 10, 1000)
XX, YY = torch.meshgrid(x, y)

batch = torch.vstack([XX.flatten(), YY.flatten()]).T
with torch.no_grad():
    ZZ = net.forward(batch).reshape(XX.shape)

with torch.no_grad():
    ZZ_icnn = icnn_net.forward(batch).reshape(XX.shape)

with torch.no_grad():
    ZZ_uncons = unconstrained_net.forward(batch).reshape(XX.shape)
ZZ_target = target_function(batch).reshape(XX.shape)

fig, axs = plt.subplots(2, 2, figsize=(6, 6))

cp = axs[0, 0].contourf(XX, YY, ZZ_target.detach(), 20, cmap='viridis',
                        label="Target")  # 20 contour levels
cp = axs[0, 1].contourf(XX, YY, ZZ_uncons.detach(), 20, cmap='viridis',
                        label="Unconstrained")  # 20 contour levels
cp = axs[1, 0].contourf(XX, YY, ZZ_icnn.detach(), 20, cmap='viridis',
                        label="ICNN")  # 20 contour levels
# axs[0].xlabel('X')
cp = axs[1, 1].contourf(XX, YY, ZZ.detach(), 20, cmap='viridis',
                        label="Ours")  # 20 contour levels
# axs[0].xlabel('X')
# axs[1].ylabel('Y')
# Tighten the layout to ensure minimal borders and whitespace
plt.tight_layout()
axs[0, 0].set_title('Target')
axs[0, 1].set_title('Unconstrained')
axs[1, 0].set_title('ICNN')
axs[1, 1].set_title('Convex Reg')
plt.savefig(results_save_path + "visu_net.pdf")
# Show the plot
plt.show()


fig, axs = plt.subplots(1, 2)

batch = torch.vstack([XX.flatten(), YY.flatten()]).T
with torch.no_grad():
    ZZ = net.forward(batch)
with torch.no_grad():
    ZZ_icnn = icnn_net.forward(batch)
ZZ_target = target_function(batch)


norm_net = torch.norm(ZZ_target - ZZ, dim=1)
norm_icnn = torch.norm(ZZ_target - ZZ_icnn, dim=1)
vmax = max(norm_icnn.max(), norm_net.max())
cp = axs[0].contourf(XX, YY, norm_icnn.detach().reshape(XX.shape), 20, cmap='viridis',
                     vmax=vmax)  # 20 contour levels
# axs[0].xlabel('X')
cp = axs[1].contourf(XX, YY, norm_net.detach().reshape(XX.shape), 20, cmap='viridis',
                     vmax=vmax)  # 20 contour levels
# axs[0].xlabel('X')
# axs[1].ylabel('Y')
# Tighten the layout to ensure minimal borders and whitespace
axs[0].set_title('ICNN')
axs[1].set_title('Convex Reg')
cbar = fig.colorbar(cp, orientation='vertical')
plt.tight_layout()
plt.savefig(results_save_path + "visu_norm_net.pdf")
# Show the plot
plt.show()

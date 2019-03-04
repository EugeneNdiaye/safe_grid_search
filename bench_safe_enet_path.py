from safegridoptim.elastic_net import enet_path as enet_solvers
from sklearn.datasets import make_regression
from sklearn.datasets.mldata import fetch_mldata
from sklearn.datasets import load_breast_cancer
from safe_enet_path import compute_path, error_grid
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy as sp
import pandas as pd
import time
import os

full_time = time.time()


def set_style():
    # This sets reasonable defaults for font size for
    # a figure that will go in a paper
    sns.set_context("paper")
    # Set the font to be serif, rather than sans
    sns.set(font='serif', font_scale=1.5)
    sns.set_palette('muted')
    # Make the background white, and specify the
    # specific font family
    sns.set_style("whitegrid", {
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"]
    })


plt.rcParams["text.usetex"] = True
set_style()
saving_fig = False  # active to True to save images

random_state = 414
dataset = "synthetic"
# dataset = "leukemia"
# dataset = "finance"
# dataset = "cancer"
# dataset = "climate"  # XXX can't be launched need, to handle auto download

run_bench = True
save_bench = False
load_bench = False
display_bench = True

if dataset == "leukemia":
    data = fetch_mldata(dataset)
    X = data.data
    y = data.target
    X = X.astype(float)
    y = y.astype(float)

if dataset == "synthetic":
    n_samples, n_features = (100, 500)
    X, y = make_regression(n_samples=n_samples, n_features=n_features,
                           random_state=random_state)

if dataset == "climate":
    X = np.load("XclimateDakar.npy")
    y = np.load("yclimateDakar.npy")
    X = X.astype(float)
    y = y.astype(float)

if dataset == "cancer":
    data = load_breast_cancer()
    X = data.data
    y = data.target

if dataset == "finance":
    X = sp.sparse.load_npz('finance_filtered.npz')
    y = np.load("finance_target.npy")

else:
    X = np.asfortranarray(X)
    # X /= np.linalg.norm(X, axis=0)
y = np.asfortranarray(y)
# The conversion to fortran type array is really important
# to avoid multiple data conversion in the optimization solver.

print("\n dataset is ", dataset, " and shape(X) = ", X.shape)
print(50 * "-", "\n")

y = (y - y.mean()) / y.std()
nrm2_y = np.linalg.norm(y) ** 2

lambda_max = np.linalg.norm(X.T.dot(y), ord=np.inf)
lambda_min = lambda_max / 1e3
n_lambdas_grid = np.array([10, 30, 50, 70, 100, 300])

algo = "Lasso"
mu = 0.
nu = 1.
tau = 10.

times = np.zeros((5, len(n_lambdas_grid)))
n_grids = np.zeros((5, len(n_lambdas_grid)), dtype=int)
errors_path = np.zeros((5, len(n_lambdas_grid)))
max_gaps = np.zeros((5, len(n_lambdas_grid)))

grid_names = ("Default grid", "Uniform unilateral", "Uniform bilateral",
              "Adaptive unilateral", "Adaptive bilateral")
grid_colors = ("#000000", "#0000CD", "#999999", "#800080", '#800000')


def bench(n_lambdas=100, eps=1e-4):

    lambda_range = lambda_min, lambda_max

    default_lambdas = lambda_max * \
        (lambda_min / lambda_max) ** (np.arange(n_lambdas) / (n_lambdas - 1.))

    tic = time.time()
    eps_ = eps * np.linalg.norm(y) ** 2
    model = enet_solvers(X, y, default_lambdas, None, mu, eps_)
    tac = time.time() - tic

    times[0, i_grid] = tac
    betas = model[1]
    gaps = model[2]
    n_grids[0, i_grid] = default_lambdas.shape[0]
    max_gaps[0, i_grid] = np.max(gaps)
    eps_path = error_grid(X, y, betas, gaps, default_lambdas, mu, nu)
    xi = eps_path
    errors_path[0, i_grid] = xi

    print("\n time default grid = ", tac, "error default = ", xi,
          "max_gap = ", np.max(gaps))

    for i_method, method in enumerate(grid_names[1:]):
        i_method += 1

        if method == "Uniform unilateral":
            adaptive = False
            large_step = False

        elif method == "Uniform bilateral":
            adaptive = False
            large_step = True

        elif method == "Adaptive unilateral":
            adaptive = True
            large_step = False

        else:  # method is "Adaptive bilateral":
            adaptive = True
            large_step = True

        tic = time.time()
        lambdas, gaps, betas = compute_path(X, y, xi, mu, nu, lambda_range,
                                            adaptive, large_step, tau)
        tac = time.time() - tic

        times[i_method, i_grid] = tac
        n_grids[i_method, i_grid] = lambdas.shape[0]
        max_gaps[i_method, i_grid] = np.max(gaps)
        error = error_grid(X, y, betas, gaps, lambdas, mu, nu)
        errors_path[i_method, i_grid] = error

        print("time = ", tac, "error = ", errors_path[i_method, i_grid],
              "n_grid = ", lambdas.shape[0], "max_gap = ", np.max(gaps))


if run_bench:

    for i_grid, n_lambdas in enumerate(n_lambdas_grid):
        bench(n_lambdas=n_lambdas, eps=1e-4)

if save_bench:

    np.save("bench_data/times_lasso_" + dataset + ".npy", times)
    np.save("bench_data/errors_path_lasso_" + dataset + ".npy", errors_path)
    np.save("bench_data/n_grids_lasso_" + dataset + ".npy", n_grids)

if load_bench:

    times = np.load("bench_data/times_lasso_" + dataset + ".npy")
    errors_path = np.load("bench_data/errors_path_lasso_" + dataset + ".npy")
    n_grids = np.load("bench_data/n_grids_lasso_" + dataset + ".npy")

if display_bench:

    times_ref = times[0, :].copy()
    relative_times = times / times[0, :]
    df = pd.DataFrame(relative_times.T, columns=grid_names)

    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, figsize=(12, 7),
                                   sharex=True)
    Title = "Computation to achieve similar precision to default "
    Title += r"$(n_{grid}, \epsilon_{opt}=10^{-4})$"

    df.plot(kind='bar', ax=ax0, rot=0, color=grid_colors, legend=True)
    ax0.legend(ncol=2)
    ax0.grid(False)
    ax0.set_ylabel("Computational time \n w.r.t. default grid")

    n_grids_ref = n_grids[0, :].copy()
    relative_n_grids = n_grids / n_grids[0, :]
    df_grid = pd.DataFrame(relative_n_grids.T, columns=grid_names)
    df_grid.plot(kind='bar', ax=ax1, rot=0, color=grid_colors, legend=False)
    ax1.set_ylabel("Grid size \n w.r.t. default grid",
                   multialignment='center')
    ax1.grid(False)

    plt.xticks(range(n_grids.shape[1]), n_grids_ref)
    ax1.set_xlabel(r"Size of the default grid")
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    if saving_fig:
        if not os.path.exists("img"):
            os.makedirs("img")
        plt.savefig("img/bench_" + algo + "_" + dataset +
                    "_grid_n_lambdas_tau" + str(int(tau)) + ".pdf",
                    format="pdf")

    print("Execution time is ", time.time() - full_time, "s")
    plt.show()

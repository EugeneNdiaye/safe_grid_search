from safegridoptim.logreg import logreg_path as logreg_solvers
from safe_logreg_path import compute_path, error_grid
from sklearn.datasets import make_classification
from sklearn.datasets.mldata import fetch_mldata
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
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

run_bench = True
save_bench = False
load_bench = False
display_bench = True

if dataset == "leukemia":
    data = fetch_mldata(dataset)
    X = data.data
    y = data.target
    y[y == -1] = 0

if dataset == "synthetic":

    n_samples, n_features = (100, 500)
    X, y = make_classification(n_samples, n_features, n_classes=2,
                               random_state=random_state)

X = np.asfortranarray(X)
y = np.asfortranarray(y)
X = X.astype(float)
y = y.astype(float)
X /= np.linalg.norm(X, axis=0)

n_samples, n_features = X.shape

print("\n dataset is ", dataset, " and shape(X) = ", X.shape)
print(50 * "-", "\n")

lambda_max = np.linalg.norm(np.dot(X.T, 0.5 - y), ord=np.inf)
lambda_min = lambda_max / 1e3
n_lambdas_grid = np.array([10, 30, 50, 70, 100, 300])

algo = "Logreg"
tau = 10.

times = np.zeros((2, len(n_lambdas_grid)))
n_grids = np.zeros((2, len(n_lambdas_grid)), dtype=int)
errors_path = np.zeros((2, len(n_lambdas_grid)))
max_gaps = np.zeros((2, len(n_lambdas_grid)))

grid_names = ("Default grid", "Adaptive unilateral")
grid_colors = ("#000000", "#800080")


def bench(n_lambdas=100, eps=1e-4):

    lambda_range = lambda_min, lambda_max

    default_lambdas = lambda_max * \
        (lambda_min / lambda_max) ** (np.arange(n_lambdas) / (n_lambdas - 1.))

    n_1 = np.sum(y == 1)
    n_0 = n_samples - n_1
    eps_ = eps * max(1, min(n_1, n_0)) / float(n_samples)

    tic = time.time()
    model = logreg_solvers(X, y, default_lambdas, None, eps=eps_)
    tac = time.time() - tic

    betas = model[0]
    gaps = model[1]

    times[0, i_grid] = tac
    n_grids[0, i_grid] = default_lambdas.shape[0]
    max_gaps[0, i_grid] = np.max(gaps)
    epsilon_path = error_grid(X, y, betas, gaps, default_lambdas)
    xi = epsilon_path
    errors_path[0, i_grid] = xi
    print("time default grid = ", tac, "error default = ", xi,
          "max_gap = ", np.max(gaps))

    tic = time.time()
    lambdas, gaps, betas = compute_path(X, y, xi, lambda_range)
    tac = time.time() - tic

    times[1, i_grid] = tac
    n_grids[1, i_grid] = lambdas.shape[0]
    max_gaps[1, i_grid] = np.max(gaps)
    error = error_grid(X, y, betas, gaps, lambdas)
    errors_path[1, i_grid] = error
    print("time = ", tac, "error = ", error,
          "n_grid = ", lambdas.shape[0], "max_gap = ", np.max(gaps))


if run_bench:
    for i_grid, n_lambdas in enumerate(n_lambdas_grid):
        bench(n_lambdas=n_lambdas, eps=1e-4)

if save_bench:
    np.save("bench_data/times_logreg_" + dataset + ".npy", times)
    np.save("bench_data/errors_path_logreg_" + dataset + ".npy", errors_path)
    np.save("bench_data/n_grids_logreg_" + dataset + ".npy", n_grids)

if load_bench:
    times = np.load("bench_data/times_logreg_" + dataset + ".npy")
    errors_path = np.load("bench_data/errors_path_logreg_" + dataset + ".npy")
    n_grids = np.load("bench_data/n_grids_logreg_" + dataset + ".npy")

if display_bench:

    times_ref = times[0, :].copy()
    relative_times = times / times[0, :]
    df = pd.DataFrame(relative_times.T, columns=grid_names)

    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, figsize=(12, 7),
                                   sharex=True)
    Title = "Computation to achieve similar precision to default "
    Title += r"$(n_{grid}, \epsilon_{opt}=10^{-4})$"
    df.plot(kind='bar', ax=ax0, rot=0, color=grid_colors, legend=False)
    ax0.grid(False)
    ax0.set_ylabel("Computational time \n w.r.t. default grid")

    n_grids_ref = n_grids[0, :].copy()
    relative_n_grids = n_grids / n_grids[0, :]
    df_grid = pd.DataFrame(relative_n_grids.T, columns=grid_names)
    df_grid.plot(kind='bar', ax=ax1, rot=0, color=grid_colors, legend=True)
    ax1.set_ylabel("Grid size \n w.r.t. default grid",
                   multialignment='center')
    plt.legend(loc='best')
    ax1.grid(False)

    plt.xticks(range(n_grids.shape[1]), n_grids[0, :])
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

import numpy as np
from sklearn.datasets import make_regression
from sklearn.datasets import make_sparse_uncorrelated
from sklearn.model_selection import train_test_split
from safegridoptim.elastic_net import enet_path as enet_solvers
from safe_enet_path import compute_path
import time
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import os


def set_style():
    # This sets reasonable defaults for font size for
    # a figure that will go in a paper
    sns.set_context("paper")
    # Set the font to be serif, rather than sans
    sns.set(font='serif', font_scale=1.5)
    sns.set_palette('muted')
    # Make the background white, and specify the
    # specific font family
    sns.set_style("ticks", {
        "font.family": "serif", 'text.usetex': True,
        "font.serif": ["Times", "Palatino", "serif"]
    })


plt.rcParams["text.usetex"] = True
set_style()
random_state = 414
saving_fig = False  # set to True to save images

# dataset = "synthetic_unco"  # Fig a
dataset = "synthetic"  # Fig b

if dataset is "synthetic":
    n_samples, n_features = (500, 5000)
    X, y = make_regression(n_samples=n_samples, n_features=n_features,
                           random_state=random_state)

if dataset is "synthetic_unco":
    n_samples, n_features = (30, 50)
    X, y = make_sparse_uncorrelated(n_samples=n_samples, n_features=n_features,
                                    random_state=random_state)

X = X.astype(float)
y = y.astype(float)
X = np.asfortranarray(X)
y = np.asfortranarray(y)

n_samples, n_features = X.shape
X = np.asfortranarray(X)
y = np.asfortranarray(y)
X /= np.linalg.norm(X, axis=0)
y = (y - y.mean()) / y.std()

X_train, X_test, y_train, y_test =\
    train_test_split(X, y, test_size=0.30, random_state=random_state)

norm_X_test = np.linalg.norm(X_test, ord=2)
lambda_max = np.linalg.norm(X_train.T.dot(y_train), ord=np.inf)
lambda_min = lambda_max / 100.
lambda_range = lambda_min, lambda_max
mu = 0.5

print("Selection by grid search with tiny grid")
n_lambdas = 200
approx_errors = np.ones(n_lambdas)
default_lambdas = lambda_max * \
    (lambda_min / lambda_max) ** (np.arange(n_lambdas) / (n_lambdas - 1.))

eps = 1e-14 * np.linalg.norm(y) ** 2
default_betas, default_gaps =\
    enet_solvers(X_train, y_train, default_lambdas, None, mu, eps)[1:3]

for i_lmd, lambda_ in enumerate(default_lambdas):
    approx_errors[i_lmd] =\
        np.linalg.norm(X_test.dot(default_betas[:, i_lmd]) - y_test)

i_best = np.argmin(approx_errors)
best_lambda = default_lambdas[i_best]
best_beta = default_betas[:, i_best]


print("Selection by safe grid search")
n_eval = 20
delta_eps = np.max(approx_errors) - np.min(approx_errors)
eps_min = delta_eps / 10.
eps_max = delta_eps * 10.

eps_vals = np.geomspace(eps_min, eps_max, n_eval)
selected_errors = np.empty(n_eval)
selected_lambdas = np.empty(n_eval)

for i_eps_val, eps_val in enumerate(eps_vals):

    nu = 1. / mu
    xi = 0.5 * mu * (eps_val / norm_X_test) ** 2

    print("Sequential computation of the path with")
    tic = time.time()
    path_lambdas, path_gaps, path_betas =\
        compute_path(X_train, y_train, xi, mu, nu, lambda_range,
                     adaptive=True, large_step=True)

    tac = time.time() - tic

    path_approx_errors = np.empty(path_lambdas.shape[0])

    for i_lmd, lambda_ in enumerate(path_lambdas):
        path_approx_errors[i_lmd] =\
            np.linalg.norm(X_test.dot(path_betas[:, i_lmd]) - y_test)

    if i_eps_val <= 1:
        i_best = np.argmin(path_approx_errors)
    else:
        tmp = np.where(path_approx_errors -
                       np.min(approx_errors) <= eps_val)[0]
        worst_best = np.max(path_approx_errors[tmp])
        i_best = np.where(path_approx_errors == worst_best)[0][0]

    lambda_selected = path_lambdas[i_best]
    beta_selected = path_betas[:, i_best]
    error_selected = np.linalg.norm(y_test - X_test.dot(beta_selected.ravel()))

    print("size_path = ", path_lambdas.shape[0],
          "error_selected = ", error_selected,
          "lambda_selected = ", lambda_selected)

    selected_errors[i_eps_val] = error_selected
    selected_lambdas[i_eps_val] = lambda_selected


colors_2 = cm.inferno_r(np.geomspace(0.01, 1, n_eval))[::-1]
colors = np.geomspace(0.01, 1, n_eval)[::-1]

plt.figure(figsize=(9, 4.4))
plt.plot(default_lambdas / lambda_max, approx_errors, color="k", lw=1,
         label="Validation curve at \n machine precision", zorder=-1)
plt.scatter(selected_lambdas / lambda_max, selected_errors, c=colors,
            marker="*", s=30, lw=2, cmap=plt.cm.get_cmap('inferno_r'),
            zorder=1)
ticks_legend = [0.1, 0.9]
cbar = plt.colorbar(ticks=ticks_legend, label=r"$\epsilon_{v}$")
cbar.set_ticklabels(["Low \n precision \n" + r"$\delta_v \times 10$",
                     "High \n precision \n" + r"$\delta_v / 10$"])

for i in range(n_eval):
    x = selected_lambdas[i] / lambda_max
    ymin = selected_errors[i] - eps_vals[i]
    ymax = selected_errors[i] + eps_vals[i]
    plt.vlines(x, ymin, ymax, colors=colors_2[i], lw=1)

plt.xticks([lambda_min / lambda_max, lambda_max / lambda_max],
           ["$\lambda_{\min}$", "$\lambda_{\max}$"])
plt.ylim((np.min(approx_errors) - delta_eps * 0.3,
          np.max(approx_errors) + delta_eps / 2.))
plt.ylabel(r"$\|y' - X'\beta^{(\lambda)}\|_2$")
plt.legend(loc="upper center")
plt.tight_layout()
if saving_fig:
    if not os.path.exists("img"):
            os.makedirs("img")
    plt.savefig("img/validation_" + dataset + ".pdf", format="pdf")
plt.show()

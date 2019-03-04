from safegridoptim.elastic_net import enet_path as enet_solvers
import matplotlib.pyplot as plt
from numpy.linalg import norm
import seaborn as sns
import numpy as np
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
    sns.set_style("whitegrid", {
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"]
    })


plt.rcParams["text.usetex"] = True
set_style()
saving_fig = False  # active to True to save images


def compute_path(X, y, lambda_range, xi, mu, nu, tau=10., conservative=True,
                 large_step=False, plot_other_side=False):
    """Docstring missing:Lasso by Default but can handle Enet."""
    n_samples, n_features = X.shape
    lambda_min, lambda_max = lambda_range

    xi_0 = xi / tau
    gap_0 = 0.
    nrm2_y = norm(y) ** 2
    norm_resid2 = nrm2_y
    dual_scale = lambda_max
    lambda_ = lambda_max

    beta = np.zeros(n_features, dtype=float, order='F')
    lambdas = [lambda_max]

    # fig, ax = plt.subplots(1)
    while lambda_ > lambda_min:

        lambda_prev = lambda_

        alpha = lambda_ / dual_scale
        R0_2 = norm_resid2 * alpha ** 2
        Delta0 = 0.5 * norm_resid2 * (1. - alpha ** 2)
        delta_opt = xi - gap_0
        tilde_delta_opt = Delta0 - gap_0

        rho_l = np.sqrt(2 * nu * R0_2 * delta_opt + tilde_delta_opt ** 2) -\
            tilde_delta_opt
        rho_l /= nu * R0_2
        rho = rho_l

        def Q(lmd, lambda_0):

            lmd_ = lmd / lambda_0
            return lmd_ * gap_0 + Delta0 * (1. - lmd_) +\
                0.5 * nu * R0_2 * (1. - lmd_) ** 2

        if large_step:

            if mu == 0:
                R0_p2 = 2 * (0.5 * norm_resid2 + 2 * gap_0 / rho_l)
            else:
                R0_p2 = 2 * mu * (0.5 * norm_resid2 + 2 * gap_0 / rho_l)

            delta_opt = xi - gap_0
            rho_r = np.sqrt(2 * nu * R0_p2 * delta_opt + gap_0 ** 2) - gap_0
            rho_r /= nu * R0_p2
            rho = (rho_l + rho_r) / (1. + rho_r)

            def tildeQ(lmd, lambda_0):

                lmd_ = lmd / lambda_0
                if mu == 0:
                    Delta0_p = np.sqrt(R0_p2 * 2 * gap_0)
                else:
                    Delta0_p = np.sqrt(R0_p2 * 2 * gap_0 / mu)

                return lmd_ * gap_0 + Delta0_p * (1. - lmd_) +\
                    0.5 * nu * R0_p2 * (1. - lmd_) ** 2

        lambda_ *= 1. - rho
        lambda_ = max(lambda_, lambda_min)
        lambdas += [lambda_]

        lmds = np.arange(lambda_, lambda_prev, 0.001)

        if large_step:
            tilde_Q_lmds = [tildeQ(lmd, lambda_prev) for lmd in lmds]
            plt.plot(lmds, tilde_Q_lmds, color='k')
            where = (tilde_Q_lmds <= xi)

        else:
            Q_lmds = [Q(lmd, lambda_prev) for lmd in lmds]
            plt.plot(lmds, Q_lmds, color='k')
            where = (Q_lmds <= xi)
            plt.fill_between(lmds, 0., Q_lmds, where=where,
                             facecolor='lightgrey', interpolate=True)

        model = enet_solvers(X, y, lambda_, beta, mu, xi_0)

        if conservative:
            gap_0 = xi_0
        else:
            gap_0 = abs(model[2][0])

        norm_resid2 = model[4][0]
        dual_scale = model[5][0]

        if large_step and plot_other_side:

            if mu == 0:
                R0_p2 = 2 * (0.5 * norm_resid2 + 2 * gap_0 / rho_l)
            else:
                R0_p2 = 2 * mu * (0.5 * norm_resid2 + 2 * gap_0 / rho_l)

            tQ_lmds_r = [tildeQ(lmd, lambda_) for lmd in lmds]

            plt.plot(lmds, tQ_lmds_r, color='k', linestyle='-.')

            Qmin = np.minimum(tQ_lmds_r, tilde_Q_lmds)
            where = (Qmin <= xi)
            plt.fill_between(lmds, 0., Qmin, where=where,
                             facecolor='lightgrey', interpolate=True)

    plt.axhline(y=xi, linestyle='--', color='k', lw=0.5)
    plt.axhline(y=xi_0, color='k', linestyle='--', lw=0.5)
    plt.axvline(x=lambda_max, color='k')
    plt.axvline(x=lambda_min, color='k')

    T_eps = len(lambdas)
    labels = []
    for t in range(T_eps):

        if t is 0:
            labels += ["$\lambda_{\max}$"]

        elif t is T_eps - 1:
            labels += ["$\lambda_{\min}$"]

        else:
            labels += ["$\lambda_" + str(t) + "$"]

    ax.set_xticklabels(labels)
    ax.set_xticks(lambdas)
    ax.set_yticklabels(["$\epsilon_{c}$", "$\epsilon$"], color='k')
    ax.set_yticks(np.array([xi_0, xi]))
    sx = 0.9 * (lambdas[-2] - lambda_min)
    plt.xlim((lambda_min - sx, lambda_max))
    plt.ylim((xi_0 / 50, 1.1 * xi))
    if not plot_other_side:
        plt.ylabel("Upper Bound of the Duality Gap")

    plt.tight_layout()

    return


if __name__ == '__main__':

    from sklearn.datasets import make_regression
    plt.close('all')
    dataset = "synthetic"
    # random_state = np.random.randint(50)
    random_state = 414
    n_samples, n_features = (30, 150)
    X, y = make_regression(n_samples=n_samples, n_features=n_features,
                           random_state=random_state)

    X = np.asfortranarray(X)
    y = np.asfortranarray(y)
    X /= np.linalg.norm(X, axis=0)
    y /= np.linalg.norm(y)

    lambda_max = np.linalg.norm(X.T.dot(y), ord=np.inf)
    lambda_min = lambda_max / 20.
    lambda_range = lambda_min, lambda_max

    # Lasso
    mu = 0.
    nu = 1.
    xi = 0.025 * np.linalg.norm(y) ** 2
    tau = 10.

    fig, ax = plt.subplots(1, figsize=(10, 4))
    compute_path(X, y, lambda_range, xi, mu, nu, tau=tau,
                 conservative=True, large_step=True, plot_other_side=True)
    if saving_fig:
        if not os.path.exists("img"):
            os.makedirs("img")
        plt.savefig("img/approximation_path_bilateral.pdf",
                    format="pdf")
    plt.show()

    fig, ax = plt.subplots(1, figsize=(10, 4))
    compute_path(X, y, lambda_range, xi, mu, nu, tau=tau,
                 conservative=True, large_step=False, plot_other_side=False)
    if saving_fig:
        if not os.path.exists("img"):
            os.makedirs("img")
        plt.savefig("img/approximation_path_unilateral.pdf",
                    format="pdf")
    plt.show()

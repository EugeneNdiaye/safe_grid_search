import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoLars
from sklearn import datasets
from safe_enet_path import compute_path
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
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"]
    })


plt.rcParams["text.usetex"] = True
set_style()
saving_fig = False  # active to True to save images


def lasso_objective(X, y, beta, lambda_):
    """Evaluate the lasso ojective (primal)."""
    loss = 0.5 * np.linalg.norm(y - X.dot(beta)) ** 2
    reg = np.linalg.norm(beta, ord=1)

    return loss + lambda_ * reg


def augmented_lars(betas, lambdas, n_step=3):
    """The Lars algorithm outputs a set of regularization parameters that
    corresponds to change point on the exact path. This function compute the
    augmented Lars path that add more (n_step) regularization parameter between
    two consecutive kink of the Lars. Since the Lars is piecewise linear
    between those kinks, the augmented_lars is computed exactly.
    """
    n_lambdas = lambdas.shape[0]
    n_features = betas.shape[0]
    n_plus_lambdas = (n_step - 1) * (n_lambdas - 1) + 1
    plus_lambdas = np.zeros(n_plus_lambdas)
    plus_betas = np.zeros((n_features, n_plus_lambdas))

    for t in range(n_lambdas - 1):

        for s in range(n_step):

            alpha_s = s / n_step
            k = t * (n_step - 1) + s

            plus_lambdas[k] = (1 - alpha_s) * lambdas[t] +\
                alpha_s * lambdas[t + 1]

            plus_betas[:, k] = (1 - alpha_s) * betas[:, t] +\
                alpha_s * betas[:, t + 1]

    plus_lambdas[n_plus_lambdas - 1] = lambdas[n_lambdas - 1]
    plus_betas[:, n_plus_lambdas - 1] = betas[:, n_lambdas - 1]

    return plus_lambdas, plus_betas


dataset = "diabetes"
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target
n_samples, n_features = X.shape

X = np.asfortranarray(X)
y = np.asfortranarray(y)
X /= np.linalg.norm(X, axis=0)
y = (y - y.mean()) / y.std()

lambda_max = np.linalg.norm(X.T.dot(y), ord=np.inf)
lambda_min = lambda_max / 50.

print("Computing regularization path using the LARS")
reg = LassoLars(alpha=lambda_min / n_samples, fit_path=True,
                fit_intercept=False)
reg.fit(X, y)

lars_betas = reg.coef_path_
lars_lambdas = reg.alphas_
lars_n_lambdas = lars_lambdas.shape[0]

# We add more point for ploting more precise lars path
lars_aug_lambdas, lars_aug_betas = augmented_lars(lars_betas, lars_lambdas, 10)
lars_aug_n_lambdas = lars_aug_lambdas.shape[0]
lars_aug_obj = np.empty(lars_aug_n_lambdas)

for t in range(lars_aug_n_lambdas):
    lars_aug_obj[t] = lasso_objective(X, y, lars_aug_betas[:, t],
                                      lars_aug_lambdas[t] * n_samples)


print("Computing approximation of the regularization path")
mu = 0.
nu = 1.
tau = 2.
xi = np.linalg.norm(y) ** 2 / 20.
lambda_range = lambda_min, lambda_max
approx_lambdas, approx_gaps, approx_betas =\
    compute_path(X, y, xi, mu, nu, lambda_range,
                 adaptive=True, large_step=False, tau=tau)

approx_n_lambdas = approx_lambdas.shape[0]
approx_obj = np.empty(2 * approx_n_lambdas - 1)
approx_aug_lambdas = np.empty(2 * approx_n_lambdas - 1)

for t in np.arange(approx_n_lambdas):
    approx_obj[2 * t] = lasso_objective(X, y, approx_betas[:, t],
                                        approx_lambdas[t])
    if t == approx_n_lambdas - 1:
        break

    approx_obj[2 * t + 1] = lasso_objective(X, y, approx_betas[:, t],
                                            approx_lambdas[t + 1])

for t in np.arange(approx_n_lambdas):

    approx_aug_lambdas[2 * t] = approx_lambdas[t]

    if t == approx_n_lambdas - 1:
        break

    approx_aug_lambdas[2 * t + 1] = approx_lambdas[t + 1]


fig, ax = plt.subplots(1, figsize=(10, 4))
label = r"Exact: $\mathcal{P}_{\lambda}(\hat \beta^{(\lambda)})$"
plt.plot(lars_aug_lambdas * n_samples, lars_aug_obj, lw=1,
         label=label, color="k")

label = r"Exact shifted:" +\
    r"$\mathcal{P}_{\lambda}(\hat \beta^{(\lambda)})+\epsilon$"
plt.plot(lars_aug_lambdas * n_samples, lars_aug_obj + xi, lw=1,
         label=label, linestyle='dashed', color="k")

plt.plot(approx_aug_lambdas, approx_obj,
         label=r"Approximated: $\mathcal{P}_{\lambda} (\beta^{(\lambda)})$",
         linestyle='-.', color="k")

plt.fill_between(lars_aug_lambdas * n_samples, lars_aug_obj + xi,
                 lars_aug_obj, facecolor='lightgrey', interpolate=True)

plt.ylabel(r"$P_{\lambda}(\beta)$")
ax.set_xticklabels(["$\lambda_{\min}$", "$\lambda_{\max}$"])
ax.set_xticks(np.array([lambda_min, lambda_max]))
plt.legend(loc="best")
plt.tight_layout()
if saving_fig:
    if not os.path.exists("img"):
        os.makedirs("img")
    plt.savefig("img/approx_lars_" + dataset + ".pdf",
                format="pdf")
plt.show()

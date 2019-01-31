import numpy as np
import scipy as sp
from .cd_logreg_fast import cd_logreg


def logreg_path(X, y, lambdas, beta_init=None, eps=1e-4, max_iter=int(1e7), f=10):
    """Compute l1-regularized logistic regression path with coordinate descent

    We solve:

    argmin_{beta} sum_{i=1}^{n} f_i(dot(x_{i}, beta)) + lambda * norm(beta, 1)
    where f_i(z) = -y_i * z + log(1 + exp(z)).

    Parameters
    ----------
    X : {array-like}, shape (n_samples, n_features)
        Training data. Pass directly as Fortran-contiguous data to avoid
        unnecessary memory duplication.

    y : ndarray, shape = (n_samples,)
        Target values : the label must be 0 and 1.

    lambdas : ndarray
        List of lambdas where to compute the models.

    beta_init : array, shape (n_features, ), optional
        The initial values of the coefficients.

    f : float, optional
        The screening rule will be execute at each f pass on the data

    eps : float, optional
        Prescribed accuracy on the duality gap.

    Returns
    -------
    betas : array, shape (n_features, n_lambdas)
        Coefficients along the path.

    dual_gaps : array, shape (n_lambdas,)
        The dual gaps at the end of the optimization for each lambda.

    lambdas : ndarray
        List of lambdas where to compute the models.

    n_iters : array-like, shape (n_lambdas,)
        The number of iterations taken by the block coordinate descent
        optimizer to reach the specified accuracy for each lambda.

    """

    if type(lambdas) != np.ndarray:
        lambdas = np.array([lambdas])

    n_lambdas = len(lambdas)
    n_samples, n_features = X.shape

    betas = np.zeros((n_features, n_lambdas))
    residuals = np.zeros((n_samples, n_lambdas))
    losses = np.zeros(n_lambdas)
    dual_scales = np.zeros(n_lambdas)
    gaps = np.ones(n_lambdas)
    n_iters = np.zeros(n_lambdas)

    y = np.asfortranarray(y, dtype=float)
    sparse = sp.sparse.issparse(X)

    if sparse:
        X_ = None
        X_data = X.data
        X_indices = X.indices
        X_indptr = X.indptr

    else:
        X_ = np.asfortranarray(X, dtype=float)
        X_data = None
        X_indices = None
        X_indptr = None

    if beta_init is None:
        beta_init = np.zeros(n_features, dtype=float, order='F')
        norm1_beta = 0.
        p_obj = n_samples * np.log(2)
        Xbeta = np.zeros(n_samples, dtype=float, order='F')
        residual = np.asfortranarray(y - 0.5)
    else:
        norm1_beta = np.linalg.norm(beta_init, ord=1)
        Xbeta = np.asfortranarray(X.dot(beta_init))
        exp_Xbeta = np.asfortranarray(np.exp(Xbeta))
        residual = np.asfortranarray(y - exp_Xbeta / (1. + exp_Xbeta))
        yTXbeta = np.dot(y, Xbeta)
        log_term = np.sum(np.log1p(exp_Xbeta))
        p_obj = -yTXbeta + log_term + lambdas[0] * norm1_beta

    XTR = np.asfortranarray(X.T.dot(residual))
    dual_scale = lambdas[0]  # True only if beta lambdas[0] ==  lambda_max

    for t in range(n_lambdas):

        gaps[t], p_obj, norm1_beta, dual_scale, n_iters[t] =\
            cd_logreg(X_, X_data, X_indices, X_indptr, y, beta_init, XTR,
                      Xbeta, residual, p_obj, norm1_beta, lambdas[t],
                      eps, dual_scale, max_iter, f, sparse=sparse)

        betas[:, t] = beta_init.copy()
        losses[t] = p_obj - lambdas[t] * norm1_beta
        residuals[:, t] = residual.copy()
        dual_scales[t] = dual_scale

        if abs(gaps[t]) > eps:

            print("warning: did not converge, t = ", t)
            print("gap = ", gaps[t], "eps = ", eps)

    return (betas, gaps, n_iters, losses, residuals, dual_scales)

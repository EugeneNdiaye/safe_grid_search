import numpy as np
import scipy.sparse as sp_sparse
from numpy.linalg import norm
from .fast_elastic_net import cd_lasso, matrix_column_norm


def enet_path(X, y, lambdas, beta_init=None, lambda2=0.5, eps=1e-4,
              max_iter=int(1e7), f=10, fit_intercept=False):
    """Compute Lasso path with coordinate descent

    The Lasso optimization solves:

    argmin_{beta} 0.5 * norm(y - X beta, 2)^2 +
                  lambda * norm(beta, 1) + 0.5 * lambda2 norm(beta, 2) ** 2

    Parameters
    ----------
    X : {array-like}, shape (n_samples, n_features)
        Training data. Pass directly as Fortran-contiguous data to avoid
        unnecessary memory duplication.

    y : ndarray, shape = (n_samples,)
        Target values

    beta_init : array, shape (n_features, ), optional
        The initial values of the coefficients.

    lambdas : ndarray
        List of lambdas where to compute the models.

    lambda_2 : float
        Second regularization parameter for the elastic net penaltys

    eps : float, optional
        Prescribed accuracy on the duality gap.

    f : float, optional
        The duality gap will be checked at each f pass on the data

    fit_intercept : bool, default value is False
        Bolean value to decide if we fit the intercept or not

    Returns
    -------
    intercepts : array, shape (n_lambdas)
        intercept value along the paths

    betas : array, shape (n_features, n_lambdas)
        Coefficients along the path.

    gaps : array, shape (n_lambdas,)
        The dual gaps at the end of the optimization for each lambda.

    n_iters : array-like, shape (n_lambdas,)
        The number of iterations taken by the block coordinate descent
        optimizer to reach the specified accuracy for each lambda.

    norm_resid2s : array, shape (n_lambdas)
        Norm of the residuals along the path

    dual_scales : array, shape (n_lambdas)
        Scaling used to compute the duality gap along the path

    """

    if type(lambdas) != np.ndarray:
        lambdas = np.array([lambdas])

    y = np.asfortranarray(y)
    n_samples, n_features = X.shape
    n_lambdas = lambdas.shape[0]

    if beta_init is None:
        beta_init = np.zeros(n_features, dtype=float, order='F')

    betas = np.zeros((n_features, n_lambdas))
    intercepts = np.zeros(n_lambdas)
    gaps = np.ones(n_lambdas)
    norm_resid2s = np.zeros(n_lambdas)
    dual_scales = np.zeros(n_lambdas)
    n_iters = np.zeros(n_lambdas)

    sparse = sp_sparse.issparse(X)
    center = fit_intercept

    if center:
        # We center the data for the intercept
        X_mean = np.asfortranarray(X.mean(axis=0)).ravel()
        y_mean = y.mean()
        y -= y_mean
        if not sparse:
            X -= X_mean
    else:
        X_mean = None

    if sparse:
        X_ = None
        X_data = X.data
        X_indices = X.indices
        X_indptr = X.indptr
        norm_Xcent = np.zeros(n_features, dtype=float, order='F')
        matrix_column_norm(n_samples, n_features, X_data, X_indices, X_indptr,
                           norm_Xcent, X_mean, center=center)
        if center:
            residual = np.asfortranarray(y - X.dot(beta_init) +
                                         X_mean.dot(beta_init))
            sum_residual = residual.sum()
        else:
            residual = np.asfortranarray(y - X.dot(beta_init))
            sum_residual = 0
    else:
        X_ = np.asfortranarray(X)
        X_data = None
        X_indices = None
        X_indptr = None
        norm_Xcent = (X_ ** 2).sum(axis=0)
        residual = np.asfortranarray(y - X.dot(beta_init))
        sum_residual = 0.

    nrm2_y = norm(y) ** 2
    XTR = np.asfortranarray(X.T.dot(residual))

    for t in range(n_lambdas):

        gaps[t], sum_residual, n_iters[t], norm_resid2s[t], dual_scales[t] = \
            cd_lasso(X_, X_data, X_indices, X_indptr, y, X_mean, beta_init,
                     norm_Xcent, XTR, residual, nrm2_y, lambdas[t], lambda2,
                     sum_residual, eps, max_iter, f, sparse=sparse,
                     center=center)

        betas[:, t] = beta_init.copy()
        if fit_intercept:
            intercepts[t] = y_mean - X_mean.dot(beta_init)

        if abs(gaps[t]) > eps:

            print("warning: did not converge, t = ", t)
            print("gap = ", gaps[t], "eps = ", eps)

    return (intercepts, betas, gaps, n_iters, norm_resid2s, dual_scales)

import numpy as np
from numpy.linalg import norm
from safegridoptim.logreg import logreg_path as logreg_solvers
from scipy import optimize


def omega(tau):
    """
    Function that appears in the upper/lower of the first order taylor
    approximation of the logistic loss
    """

    if 1. <= tau and tau <= 1. + 1e-12:
        return 1.

    elif abs(tau) <= 1e-12:
        return 0.5

    return ((1. - tau) * np.log(1. - tau) + tau) / (tau ** 2)


def step_size(y, residual, loss_Xbeta, eps, eps_c, lambda_, dual_scale):
    """Compute adaptive unilateral step size for the eps-approximation path.
       Note that, for logistic regression, the adaptive bilateral is harder
       to compute. It requires B_f (a constant that appears in the complexity).

    Parameters
    ----------

    y : ndarray, shape = (n_samples,)
        Target values

    residual : ndarray, shape = (n_samples,)
        Residual value at the current parameter lambda_

    loss_Xbeta : float
        Loss function value at the current parameter lambda_

    eps : float
        Desired accuracy on the whole path

    eps_c : float
        Optimization accuracy at each step, it must satisfies eps_c < eps

    lambda_ : float
        Current regularization parameter

    dual_scale : float
        Scaling used to make the residual dual feasible

    Returns
    -------
    rho : float
        Step size for computing the next regularization parameter from lambda_
    """

    _lambda_theta = -residual * (lambda_ / dual_scale)
    tmp = _lambda_theta + y
    nabla_dual = np.log(tmp) - np.log(1. - tmp)  # np.log(tmp / (1. - tmp))
    loss_nabla_dual = (-np.dot(y.T, nabla_dual) +
                       np.sum(np.log1p(np.exp(nabla_dual)), axis=0))
    # Todo: find a way to avoid recomputation in loss_nabla_dual
    # (lambda_ / dual_scale) == 1 almost all the time

    K_0 = loss_Xbeta - loss_nabla_dual
    norm_zeta = norm(_lambda_theta)

    # For computational efficiency, we do not build the diagonal matrix
    # hessian_dual are equal to np.diag(1. / (tmp * (1. - tmp)))

    hessian = 1. / (tmp * (1. - tmp))
    norm_hess_zeta2 = np.sum(hessian * _lambda_theta ** 2)
    ratio_norm_term = norm_hess_zeta2 / norm_zeta

    def Q_(rho):

        rho_hess_det = norm_hess_zeta2 * rho ** 2
        return (eps_c + rho * (K_0 - eps_c) +
                omega(ratio_norm_term * rho) * rho_hess_det - eps)

    def QQ(rho):

        rho_hess_det = norm_hess_zeta2 * rho ** 2
        return (eps_c + rho * (K_0 - eps_c) + rho_hess_det - eps)

    if np.sign(Q_(1e-50)) == np.sign(Q_(1. / ratio_norm_term)):
        rho = 1. / ratio_norm_term

    else:
        rho = optimize.brentq(Q_, 0., 1. / ratio_norm_term, xtol=1e-20)

    # TODO: implement left size

    return rho


def compute_path(X, y, eps, lambda_range, tau=10.):
    """Compute an eps-approximation path on a given range of parameter

    Parameters
    ----------
    X : {array-like}, shape (n_samples, n_features)
        Training data. Pass directly as Fortran-contiguous data to avoid
        unnecessary memory duplication in the optimization solver.

    y : ndarray, shape = (n_samples,)
        Target values

    eps : float
        Desired accuracy on the whole path

    lambda_range : list or tuples, or array of size 2
        Range of parameter where the eps-path is computed.
        lambda_range = [lambda_min, lambda_max]

    tau : float, optional (strictly larger than 1)
        At each step, we solve an optimization problem at accuracy eps_c < eps.
        We simply choose eps_c = eps / tau. Default value is tau=10.

    Returns
    -------
    lambdas : ndarray
        List of lambdas that constitutes the eps-path grid

    gaps : array, shape (n_lambdas,)
        The dual gaps at the end of the optimization for each lambda.

    betas : array, shape (n_features, n_lambdas)
        Coefficients beta along the eps-path.

    """

    n_samples, n_features = X.shape
    lambda_min, lambda_max = lambda_range

    loss_Xbeta = n_samples * np.log(2)
    residual = np.asfortranarray(y - 0.5)

    lambda_ = lambda_max
    lambdas = [lambda_]
    path_gaps = [0.]
    dual_scale = lambda_max
    eps_c = eps / tau
    beta = np.zeros(n_features)

    # decreasing direction of lambda_
    betas = [beta.copy()]
    while lambda_ > lambda_min:

        # Update lambda_
        # In step size(), one can use eps_c = gap
        rho = step_size(y, residual, loss_Xbeta, eps, eps_c, lambda_,
                        dual_scale)

        lambda_ *= 1. - rho
        lambda_ = max(lambda_, lambda_min)  # stop at lambda_min
        lambdas += [lambda_]

        # Update primal / dual
        model = logreg_solvers(X, y, lambda_, beta, eps_c)
        loss_Xbeta = model[3][0]
        residual = model[4][:, 0]
        dual_scale = model[5][0]
        betas += [beta.copy()]
        gap = abs(model[1][0])
        path_gaps += [gap]

    # TODO: implement it for increasing direction of lambda_  and
    # factorize the code for the two direction.

    return np.array(lambdas), np.array(path_gaps), np.array(betas).T


def Q(lambda_0, lambda_, eps_c, K_0, norm_hess_zeta2, ratio_norm_term):
    """
    Function upper bound of the duality gap function initialized at lambda_0
    """

    lmd = lambda_ / lambda_0
    Q_lambda = (lmd * eps_c + K_0 * (1. - lmd) +
                omega(ratio_norm_term * (1. - lmd)) *
                norm_hess_zeta2 * (1. - lmd) ** 2)

    return Q_lambda


def error_grid(X, y, betas, gaps, lambdas):
    """
    Compute the error eps such that the set of betas on the given grid of
    regularization parameter lambdas is an eps-path
    """

    n_samples, n_features = X.shape
    n_lambdas = lambdas.shape[0]
    K_0s = np.zeros(n_lambdas)
    norm_hess_zeta2s = np.zeros(n_lambdas)
    ratio_norm_terms = np.zeros(n_lambdas)
    Q_is = []

    for i_lambda, lambda_ in enumerate(lambdas):

        Xbeta = X.dot(betas[:, i_lambda])
        exp_Xbeta = np.exp(Xbeta)
        residual = np.asfortranarray(y - exp_Xbeta / (1. + exp_Xbeta))
        XTR = X.T.dot(residual)
        dual_scale = max(lambda_, norm(XTR, ord=np.inf))
        loss_Xbeta = -np.dot(y.T, Xbeta) + np.sum(np.log1p(exp_Xbeta), axis=0)
        _lambda_theta = -residual * (lambda_ / dual_scale)
        tmp = _lambda_theta + y
        nabla_dual = np.log(tmp) - np.log(1. - tmp)  # np.log(tmp / (1. - tmp))
        loss_nabla_dual = (-np.dot(y.T, nabla_dual) +
                           np.sum(np.log1p(np.exp(nabla_dual)), axis=0))
        K_0s[i_lambda] = loss_Xbeta - loss_nabla_dual
        norm_zeta = norm(_lambda_theta)

        # For computational efficiency, we do not build the diagonal matrix
        # hessia_dual are equal to np.diag(1. / (tmp * (1. - tmp)))

        hessian = 1. / (tmp * (1. - tmp))
        norm_hess_zeta2 = np.sum(hessian * _lambda_theta ** 2)
        norm_hess_zeta2s[i_lambda] = norm_hess_zeta2
        ratio_norm_terms[i_lambda] = norm_hess_zeta2 / norm_zeta

    for i in range(n_lambdas - 1):

        def Q_diff(lambda_):

            Q_ip = Q(lambdas[i + 1], lambda_, abs(gaps[i + 1]), K_0s[i + 1],
                     norm_hess_zeta2s[i + 1], ratio_norm_terms[i + 1])

            Q_i = Q(lambdas[i], lambda_, abs(gaps[i]), K_0s[i],
                    norm_hess_zeta2s[i], ratio_norm_terms[i])

            return Q_ip - Q_i

        const_i = lambdas[i] * (1. - 1. / ratio_norm_terms[i])
        const_ip = lambdas[i + 1] * (1. - 1. / ratio_norm_terms[i + 1])
        min_c = max(const_i, const_ip)

        if (min_c >= lambdas[i + 1] and
                np.sign(Q_diff(min_c)) != np.sign(Q_diff(lambdas[i]))):

            hat_lmbd_i = optimize.brentq(Q_diff, min_c, lambdas[i], xtol=1e-20)
            Q_is += [Q(lambdas[i], hat_lmbd_i, abs(gaps[i]), K_0s[i],
                       norm_hess_zeta2s[i], ratio_norm_terms[i])]

        elif min_c < lambdas[i + 1] and \
                np.sign(Q_diff(lambdas[i + 1])) != np.sign(Q_diff(lambdas[i])):

            hat_lmbd_i = optimize.brentq(Q_diff, lambdas[i + 1], lambdas[i],
                                         xtol=1e-20)
            Q_is += [Q(lambdas[i], hat_lmbd_i, abs(gaps[i]), K_0s[i],
                       norm_hess_zeta2s[i], ratio_norm_terms[i])]

        else:
            hat_lmbd_i = const_i
            Q_is += [Q(lambdas[i + 1], hat_lmbd_i, abs(gaps[i]), K_0s[i + 1],
                       norm_hess_zeta2s[i + 1], ratio_norm_terms[i + 1])]

    return np.max(Q_is)

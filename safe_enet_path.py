import numpy as np
from numpy.linalg import norm
from safegridoptim.elastic_net import enet_path as enet_solvers


def step_size(eps, eps_c, lambda_, norm_resid2, dual_scale, mu=0., nu=1.,
              large_step=True):
    """Compute adaptive step size for the eps-approximation path

    Parameters
    ----------
    eps : float
        Desired accuracy on the whole path

    eps_c : float
        Optimization accuracy at each step, it must satisfies eps_c < eps

    lambda_ : float
        Current regularization parameter

    norm_resid2 : float
        Squared norm of the residual at the current parameter lambda_

    dual_scale : float
        Scaling used to make the residual dual feasible

    mu : float, optional
        Strong convexity paramerer of the loss.
        Default value is 0. It corresponds to vanilla least squares

    nu : float, optional
        Smoothness paramerer of the loss.
        Default value is 1. It corresponds to vanilla least squares

    large_step : boolean, optional
        If True, it computes the bilateral step size which is larger than the
        unilateral step size (False)

    Returns
    -------
    rho : float
        Step size for computing the next regularization parameter from lambda_
    """

    alpha = lambda_ / dual_scale
    norm_zeta2 = norm_resid2 * alpha ** 2
    Delta = 0.5 * norm_resid2 * (1. - alpha ** 2)
    delta_opt = eps - eps_c
    tilde_delta_opt = Delta - eps_c

    # dir_ is "left":
    rho_l = np.sqrt(2 * nu * norm_zeta2 * delta_opt + tilde_delta_opt ** 2) -\
        tilde_delta_opt
    rho_l /= nu * norm_zeta2
    rho = rho_l

    if large_step:

        if mu == 0:
            tilde_R2 = 2 * (0.5 * norm_resid2 + 2 * eps_c / rho_l)
        else:
            tilde_R2 = 2 * mu * (0.5 * norm_resid2 + 2 * eps_c / rho_l)
        delta_opt = eps - eps_c
        rho_r = np.sqrt(2 * nu * tilde_R2 * delta_opt + eps_c ** 2) - eps_c
        rho_r /= nu * tilde_R2
        rho = (rho_l + rho_r) / (1. + rho_r)

    # TODO: add implementation of both direction (increasing and decreasing)

    return rho


def compute_path(X, y, eps, mu, nu, lambda_range, adaptive=True,
                 large_step=False, tau=10.):
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

    mu : float, optional
        Strong convexity paramerer of the loss.
        Default value is 0. It corresponds to vanilla least squares

    nu : float, optional
        Smoothness paramerer of the loss.
        Default value is 1. It corresponds to vanilla least squares

    lambda_range : list or tuples, or array of size 2
        Range of parameter where the eps-path is computed.
        lambda_range = [lambda_min, lambda_max]

    adaptive : boolean, optional
        If True (default value), the eps-path is constructed with unilateral
        or bilateral step size. If False, the uniform grid is constructed.

    large_step : boolean, optional
        If True, it computes the bilateral step size which is larger than the
        unilateral step size (False)

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

    lambda_ = lambda_max
    lambdas = [lambda_]
    gaps = [0.]
    nrm2_y = norm(y) ** 2
    norm_resid2 = nrm2_y
    dual_scale = lambda_max
    eps_c = eps / tau

    beta = np.zeros(n_features, dtype=float, order='F')

    # uniform step size
    if adaptive is False:

        nu_nrm2_y = nu * nrm2_y
        if mu == 0:
            tmp = np.sqrt(2. * nrm2_y * eps_c) - eps_c
        else:
            tmp = np.sqrt(2. * mu * nrm2_y * eps_c) - eps_c
        delta_opt = eps - eps_c
        rho_l = np.sqrt(2. * nu_nrm2_y * delta_opt + tmp ** 2) - tmp
        rho_l /= nu_nrm2_y

        if large_step is False:
            rho = rho_l

        else:
            rho_r = np.sqrt(2. * nu_nrm2_y * delta_opt + eps_c ** 2) - eps_c
            rho_r /= nu_nrm2_y
            rho = (rho_l + rho_r) / (1. + rho_r)

        r_lmbd = lambda_min / lambda_max
        n_lambdas = int(1 + np.floor(np.log(r_lmbd) / np.log(1. - rho)))
        lambdas = lambda_max * (1. - rho) ** np.arange(n_lambdas)
        model = enet_solvers(X, y, lambdas, beta, mu, eps_c)
        betas = model[1]
        gaps = model[2]

        return lambdas, gaps, betas

    # Adaptive grid with decreasing direction of lambda_
    betas = [beta.copy()]
    while lambda_ > lambda_min:

        # Update lambda_
        # In step size(), one can use eps_c = gap
        rho_l = step_size(eps, eps_c, lambda_, norm_resid2, dual_scale, mu, nu,
                          large_step)
        lambda_ *= 1. - rho_l
        lambda_ = max(lambda_, lambda_min)  # stop at lambda_min
        lambdas += [lambda_]

        model = enet_solvers(X, y, lambda_, beta, mu, eps=eps_c)
        norm_resid2 = model[4][0]
        dual_scale = model[5][0]
        betas += [beta.copy()]
        gap = abs(model[2][0])
        gaps += [gap]

    # TODO: implement it for increasing direction of lambda_  and
    # factorize the code for the two direction.

    return np.array(lambdas), np.array(gaps), np.array(betas).T


def Q(lambda_0, lambda_, eps_c, Delta, norm_zeta2, nu):
    """
    Quadratic upper bound of the duality gap function initialized at lambda_0
    """

    lmd = lambda_ / lambda_0
    Q_lambda = (lmd * eps_c + Delta * (1. - lmd) +
                0.5 * nu * norm_zeta2 * (1. - lmd) ** 2)

    return Q_lambda


def error_grid(X, y, betas, gaps, lambdas, mu, nu):
    """
    Compute the error eps such that the set of betas on the given grid of
    regularization parameter lambdas is an eps-path
    """

    n_samples, n_features = X.shape
    n_lambdas = lambdas.shape[0]
    Deltas = np.zeros(n_lambdas)
    norm_zeta2s = np.zeros(n_lambdas)
    Q_is = []

    for i, lambda_ in enumerate(lambdas):

        Xbeta = X.dot(betas[:, i])
        residual = y - Xbeta
        XTR = X.T.dot(residual)
        norm_beta2 = norm(betas[:, i]) ** 2
        norm_resid2 = norm(residual) ** 2 + mu * norm_beta2
        dual_scale = max(lambda_, norm(XTR - mu * betas[:, i], ord=np.inf))
        alpha = lambda_ / dual_scale
        norm_zeta2s[i] = norm_resid2 * alpha ** 2
        Deltas[i] = abs(0.5 * norm_resid2 * (1. - alpha ** 2))
        gaps[i] = abs(gaps[i])  # avoid negative gap due to numerical errors

    for i in range(n_lambdas - 1):

        a = norm_zeta2s[i + 1] / (lambdas[i + 1] ** 2) - \
            norm_zeta2s[i] / (lambdas[i] ** 2)
        a *= 0.5 * nu

        diff_gap = gaps[i + 1] / lambdas[i + 1] - gaps[i] / lambdas[i]
        diff_Delta = Deltas[i + 1] / lambdas[i + 1] - Deltas[i] / lambdas[i]
        diff_norm_zeta2 = norm_zeta2s[i + 1] / \
            lambdas[i + 1] - norm_zeta2s[i] / lambdas[i]

        b = diff_gap - diff_Delta - nu * diff_norm_zeta2
        c = Deltas[i + 1] - Deltas[i] + 0.5 * nu * \
            (norm_zeta2s[i + 1] - norm_zeta2s[i])

        if abs(a) >= 1e-12:
            discriminant = b ** 2 - 4 * a * c
            hat_lmbdi = (np.sqrt(discriminant) - b) / (2. * a)
        else:
            if abs(b) >= 1e-12:
                hat_lmbdi = - c / b
            else:
                hat_lmbdi = -666

        if hat_lmbdi == -666:
            Q_is += [Q(lambdas[i + 1], lambdas[i], gaps[i + 1], Deltas[i + 1],
                       norm_zeta2s[i + 1], nu)]
        else:
            Q_is += [Q(lambdas[i], hat_lmbdi, gaps[i],
                       Deltas[i], norm_zeta2s[i], nu)]

    return np.max(Q_is)

# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from libc.math cimport fabs, sqrt
from scipy.linalg.cython_blas cimport daxpy, ddot, dnrm2, dscal
import numpy as np
cimport numpy as np
cimport cython

cdef int inc = 1  # Default array increment for cython_blas operation


cdef inline double fmax(double x, double y) nogil:
    if x > y:
        return x
    return y


cdef inline double fsign(double f) nogil:
    if f == 0:
        return 0
    elif f > 0:
        return 1.0
    else:
        return -1.0


cdef double abs_max(int n, double * a) nogil:
    """np.max(np.abs(a))"""
    cdef int i
    cdef double m = fabs(a[0])
    cdef double d
    for i in range(1, n):
        d = fabs(a[i])
        if d > m:
            m = d
    return m


cdef double max(int n, double * a) nogil:
    """np.max(a)"""
    cdef int i
    cdef double m = a[0]
    cdef double d
    for i in range(1, n):
        d = a[i]
        if d > m:
            m = d
    return m


def matrix_column_norm(int n_samples, int n_features, double[::1] X_data,
                       int[::1] X_indices, int[::1] X_indptr,
                       double[::1] norm_Xcent, double[::1] X_mean,
                       int center=0):
    cdef:
        int i = 0
        int j = 0
        int i_ptr = 0
        int start = 0
        int end = 0
        double X_mean_j = 0.

    with nogil:
        for j in range(n_features):

            if center:
                X_mean_j = X_mean[j]

            start = X_indptr[j]
            end = X_indptr[j + 1]
            norm_Xcent[j] = 0.

            for i_ptr in range(start, end):
                norm_Xcent[j] += (X_data[i_ptr] - X_mean_j) ** 2

            if center:
                norm_Xcent[j] += (n_samples - end + start) * X_mean_j ** 2


cdef double primal_value(int n_features, double * beta_data,
                         double norm_resid2, double lambda_,
                         double lambda2) nogil:

    cdef:
        double l1_norm = 0
        int j = 0

    for j in range(n_features):
        l1_norm += fabs(beta_data[j])

    return 0.5 * norm_resid2 + lambda_ * l1_norm


cdef double dual(int n_samples, int n_features, double * residual_data,
                 double * y_data, double dual_scale, double norm_resid2,
                 double lambda_) nogil:

    cdef:
        double dval = 0.
        double alpha = lambda_ / dual_scale
        double Ry = ddot(& n_samples, residual_data, & inc, y_data, & inc)

    if dual_scale != 0:
        dval = alpha * Ry - 0.5 * alpha ** 2 * norm_resid2

    return dval


cdef double ST(double u, double x) nogil:
    return fsign(x) * fmax(fabs(x) - u, 0)


cdef void update_residual(int n_samples, double * X_j, double[::1] X_data,
                          int[::1] X_indices, double * residual,
                          double * beta, double X_mean_j, int startptr,
                          int endptr, double beta_diff, double * sum_residual,
                          int sparse, int center) nogil:

    cdef:
        int i = 0
        int i_ptr = 0

    if sparse:
        for i_ptr in range(startptr, endptr):
            i = X_indices[i_ptr]
            residual[i] += X_data[i_ptr] * beta_diff

        if center:
            sum_residual[0] = 0.
            for i in range(n_samples):
                residual[i] -= X_mean_j * beta_diff
                sum_residual[0] += residual[i]
    else:
        daxpy(& n_samples, & beta_diff, X_j, & inc, & residual[0], & inc)


cdef void compute_XTR_j(int n_samples, double * XTR_j, double * X_j_ptr,
                        double[::1] X_data, int[::1] X_indices, int startptr,
                        int endptr, double * residual, double X_mean_j,
                        double sum_residual, double lambda2, double beta_j,
                        int sparse, int center) nogil:

    cdef:
        int i = 0
        int i_ptr = 0
    XTR_j[0] = 0.

    if sparse:

        for i_ptr in range(startptr, endptr):
            i = X_indices[i_ptr]
            XTR_j[0] += X_data[i_ptr] * residual[i]

        if center:
            XTR_j[0] -= X_mean_j * sum_residual

    else:
        XTR_j[0] = ddot(& n_samples, X_j_ptr, & inc, & residual[0], & inc)

    XTR_j[0] -= lambda2 * beta_j


def cd_lasso(double[::1, :] X, double[::1] X_data, int[::1] X_indices,
             int[::1] X_indptr, double[::1] y, double[::1] X_mean,
             double[::1] beta, double[::1] norm_Xcent, double[::1] XTR,
             double[::1] residual, double nrm2_y, double lambda_,
             double lambda2, double sum_residual, double eps,
             int max_iter, int f, int sparse=0, int center=0):
    """
        Solve 1/2 ||y - X beta||^2 + lambda_ ||beta||_1
    """

    cdef:
        int i = 0
        int j = 0
        int i_ptr = 0
        int startptr = 0
        int endptr = 0

        int n_iter = 0
        int n_samples = y.shape[0]
        int n_features = beta.shape[0]

        double gap_t = 1
        double double_tmp = 0
        double beta_old_j = 0
        double p_obj = 0.
        double d_obj = 0.
        double r_normX_j = 0.
        double X_mean_j = 0.
        double beta_diff = 0.
        double r_screen = 1.
        double norm_resid2 = 1.
        double dual_scale = 0.
        double norm_beta2 = 0.
        double L_j = 0.
        double mu_j = 0.
        double * X_j_ptr = & X[0, 0]

    with nogil:

        for n_iter in range(max_iter):

            if f != 0 and n_iter % f == 0:

                # Computation of XTR
                double_tmp = 0.
                # theta_k = residual / dual_scale
                for j in range(n_features):

                    if sparse:
                        startptr = X_indptr[j]
                        endptr = X_indptr[j + 1]
                        if center:
                            X_mean_j = X_mean[j]
                    else:
                        X_j_ptr = & X[0, j]

                    compute_XTR_j(n_samples, & XTR[j], X_j_ptr, X_data,
                                  X_indices, startptr, endptr, & residual[0],
                                  X_mean_j, sum_residual, lambda2, beta[j],
                                  sparse, center)

                    double_tmp = fmax(double_tmp, fabs(XTR[j]))

                dual_scale = fmax(lambda_, double_tmp)
                norm_beta2 = dnrm2(& n_features, & beta[0], & inc) ** 2
                norm_resid2 = dnrm2(& n_samples, & residual[0], & inc) ** 2 + lambda2 * norm_beta2

                p_obj = primal_value(n_features, & beta[0], norm_resid2,
                                     lambda_, lambda2)

                d_obj = dual(n_samples, n_features, & residual[0], & y[0],
                             dual_scale, norm_resid2, lambda_)
                gap_t = p_obj - d_obj

                if gap_t <= eps:
                    break

            # Coordinate descent
            for j in range(n_features):

                L_j = norm_Xcent[j] + lambda2
                mu_j = lambda_ / L_j
                beta_old_j = beta[j]

                if sparse:
                    startptr = X_indptr[j]
                    endptr = X_indptr[j + 1]
                    if center:
                        X_mean_j = X_mean[j]
                else:
                    X_j_ptr = & X[0, j]

                compute_XTR_j(n_samples, & XTR[j], X_j_ptr, X_data,
                              X_indices, startptr, endptr, & residual[0],
                              X_mean_j, sum_residual, lambda2, beta[j], sparse,
                              center)

                beta[j] = ST(mu_j, beta[j] + XTR[j] / L_j)

                if beta[j] != beta_old_j:

                    beta_diff = beta_old_j - beta[j]
                    update_residual(n_samples, X_j_ptr, X_data, X_indices,
                                    & residual[0], & beta[0], X_mean_j,
                                    startptr, endptr, beta_diff,
                                    & sum_residual, sparse, center)

    return (gap_t, sum_residual, n_iter, norm_resid2, dual_scale)

import numpy as np


def alpha_n(Vi, params):
    c1, c2, c3, c4 = params
    result_alpha = (c1 * (Vi + c2)) / (1 - np.exp(-c3 * (Vi + c4)))
    return result_alpha

def beta_n(Vi, params):
    c5, c6, c7 = params
    result_beta = c5 * (np.exp(-c6 * ( Vi + c7 )))
    return result_beta

def n_inf(alpha, beta):
    result_n_inf =  alpha / (alpha + beta)
    return result_n_inf

def tau_n(alpha, beta):
    result_tau = 1 / (alpha + beta)
    return result_tau

def n_pow_4(n, params):
    c8 = params
    result_n =  n ** 4
    return result_n

def get_y_hat(params, ti, Vi):
    epsilon = 0.0001
    alpha = alpha_n(Vi, params[:4])
    beta = beta_n(Vi, params[4:7])
    n_inf_ = n_inf(alpha, beta)
    tau_n_ = tau_n(alpha, beta)
    n = n_inf_ * (1 - np.exp((-ti + epsilon) / tau_n_))
    y_hat = n_pow_4(n, params[-1])
    return y_hat

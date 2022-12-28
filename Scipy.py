import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler


def alpha_n(V, params):
    c1, c2, c3, c4 = params
    result_alpha = (c1 * (V + c2)) / (1 - np.exp(-c3 * (V + c4)))
    return result_alpha

def beta_n(V, params):
    c5, c6, c7 = params
    result_beta = c5 * (np.exp(-c6 * ( V + c7 )))
    return result_beta

def n_inf(alpha, beta):
    result_n_inf =  alpha / (alpha + beta)
    return result_n_inf

def tau_n(alpha, beta):
    result_tau = 1 / (alpha + beta)
    return result_tau

def n_pow_4(n):
    result_n =  n ** 4
    return result_n

def get_y_hat(params, t, V):
    alpha = alpha_n(V, params[:4])
    beta = beta_n(V, params[4:])
    n_inf_ = n_inf(alpha, beta)
    tau_n_ = tau_n(alpha, beta)
    n = n_inf_ * (1 - np.exp(-t/tau_n_))
    y_hat = n_pow_4(n)
    return y_hat

def nll_loss(params, t, V, y):
    epsilon = 0.01
    y_hat = get_y_hat(params, t, V)
    y_hat = np.maximum(y_hat, epsilon)
    loss = -np.mean(y * np.log(y_hat)) / float(y_hat.shape[0])
    print(loss)
    return loss 

def l2_loss(params, t, V, y):
    y_hat = get_y_hat(params, t, V)
    loss = np.mean((y_hat - y) ** 2)
    print(loss)
    return loss

def l1_loss(params, t, V, y):
    y_hat = get_y_hat(params, t, V)
    loss = np.mean(abs(y_hat - y))
    print(loss)
    return loss

def get_data(path):
    data = pd.read_csv(path)
    inputs = data.iloc[:, :-1]
    t, V = inputs
    t = inputs.iloc[:,:-1].values
    V = inputs.iloc[:,-1].values
    labels = data.iloc[:,-1].values
    return t, V, labels


def get_scaled_data(path):
    data = pd.read_csv(path)
    data = data.values
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data)
    inputs = data_standardized[:, :-1]
    labels = data_standardized[:, -1]
    t = inputs[:, :-1]
    V = inputs[:, -1]
    return t, V, labels

#t, V, labels = get_data('Prod/dataset.csv')
t, V, labels = get_scaled_data('Prod/dataset.csv')

#params_init = [0.02, 35, 0.6, 0.6, 0.125, 0.015, 55]
params_init = np.random.randn(7)

#bounds = [(-3, 3), (-100, 100), (-3, 3), (-100, 100), (-3, 3), (-3, 3), (-100, 100)]
bounds = [(-100, 100), (-100, 100), (-100, 100), (-100, 100), (-100, 100), (-100, 100), (-100, 100)]

# BFGS CG L-BFGS-B Newton-CG TNC Nelder-Mead Powell COBYLA SLSQP trust-constr dogleg trust-ncg trust-exact trust-krylov trust-constr-krylov
result = minimize(nll_loss, params_init, args=(t, V, labels), method='TNC', bounds=bounds)

print(result.x)

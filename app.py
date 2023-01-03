import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ypstruct import structure
import ga
from sklearn.preprocessing import StandardScaler

easy_varmin = [-100,  -100, -100, -100, -100, -100, -100,  4]
easy_varmax = [ 100,  100,  100, 100,  100,  100, 100, 4]
medium_varmin = [-1,    0, -1,   0, -1, -1,   0,  4]
medium_varmax = [ 1,  100,  1, 100,  1,  1, 100, 4]
extreme_varmin = [0.009, 54.9, 0.09, 54.9,  0.124, 0.0124, 64.9, 4]
extreme_varmax = [0.011,  55.1,  0.11, 55.1, 0.126, 0.0126, 65.1, 4]


easy_gamma = [[-0.001, 0.001], [-0.01, 0.01], [-0.001, 0.001], [-0.01, 0.01], [-0.001, 0.001], [-0.001, 0.001], [-0.01, 0.01], [-0.001, 0.001]]
medium_gamma = [[-0.01, 0.01], [-0.1, 0.1], [-0.01, 0.01], [-0.1, 0.1], [-0.01, 0.01], [-0.01, 0.01], [-0.1, 0.1], [-0.01, 0.01]]
extreme_gamma = [[-0.1, 0.1], [-1, 1], [-0.1, 0.1], [-1, 1], [-0.1, 0.1], [-0.1, 0.1], [-1, 1], [-0.1, 0.1]]

def get_data(path):
    data = pd.read_csv(path)
    inputs = data.iloc[:, :-1]
    t, V = inputs
    t = inputs.iloc[:,0].values
    V = inputs.iloc[:,1].values
    labels = data.iloc[:,2].values
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
    #print(y_hat[0])
    return y_hat

def l2_loss(params):
    loss = 0
    for i in range(len(t)):
        y_hat = np.round(get_y_hat(params, t[i], V[i]), 8)
        if y_hat == float("inf") or np.isnan(y_hat):
            y_hat = np.random.uniform(1,2) * 100
        loss += (np.round(y_hat,8) - np.round(labels[i],8)) ** 2
    if loss == float("inf") or np.isnan(loss): loss = np.random.uniform(1,2) * 1000
    ret_loss = loss / len(t)
    return ret_loss

def l1_loss(params):
    loss = 0
    for i in range(len(t)):
        y_hat = get_y_hat(params, t[i], V[i])
        if y_hat == float("inf") or np.isnan(y_hat):
            y_hat = np.random.uniform(1,2) * 100
        loss += np.abs(y_hat - labels[i])
    if loss == float("inf") or np.isnan(loss): loss = np.random.uniform(1,2) * 1000
    ret_loss = loss / len(t)
    return ret_loss

# Problem Definition
problem = structure()
t, V, labels = get_data('Prod/dataset.csv')
problem.t = t
problem.V = V
problem.labels = labels
problem.costfunc = l2_loss
problem.nvar = 8
problem.varmin = medium_varmin
problem.varmax = medium_varmax
problem.update_vec = easy_gamma

# GA Parameters
params = structure()
params.maxit = 500
params.npop = 200
params.beta = 1
params.pc = 1
params.gamma = 0.1
params.mu = 1
params.sigma = 0.2

# Run GA
out = ga.run(problem, params)
print(out.bestsol)

# Results
plt.plot(out.bestcost)
# plt.semilogy(out.bestcost)
plt.xlim(0, params.maxit)
plt.xlabel('Iterations')
plt.ylabel('Best Cost')
plt.title('Genetic Algorithm (GA)')
plt.grid(True)
plt.show()


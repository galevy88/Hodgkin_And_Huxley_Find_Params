import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def alpha_n(V):
    result_alpha = (0.01 * (V + 55)) / (1 - np.exp(-0.1 * (V + 55)))
    return result_alpha

def beta_n(V):
    result_beta = 0.125 * (np.exp(-0.0125 * ( V + 65 )))
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
    
def get_y(t, V):
    alpha = alpha_n(V)
    beta = beta_n(V)
    n_inf_ = n_inf(alpha, beta)
    tau_n_ = tau_n(alpha, beta)
    n = n_inf_ * (1 - np.exp(-t/tau_n_))
    y_hat = n_pow_4(n)
    return y_hat


dataset = []
n = 0
t_total = 100
VOLTS = np.arange(-90, 90, 10)
results = []

for V in VOLTS:
    voltage_results = []
    for t in np.arange(0, t_total):
        y = get_y(t, V)
        voltage_results.append(y)
        dataset.append((t, V, y))
    results.append(voltage_results)


graph_df = pd.DataFrame(results)
graph_df = graph_df.T
dataset_df = pd.DataFrame(dataset)
print(dataset_df)
graph_df.plot()
plt.legend(np.arange(-90, 90, 10))
plt.xlabel("Time (ms)")
plt.show()

# graph_df.to_csv('Prod/graph_df.csv', index=False)
# dataset_df.to_csv('Prod/dataset.csv', index=False)

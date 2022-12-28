import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def alpha_n(V, T):
    return (0.01 * (V + 55)) / (1 - np.exp(-0.1 * (V + 55)))


def beta_n(V, T):
    return 0.125 * (np.exp(-0.0125 * ( V + 65 )))


def n_inf(V, T):
    return alpha_n(V, T) / (alpha_n(V, T) + beta_n(V, T))


def tau_n(V, T):
    return 1 / (alpha_n(V, T) + beta_n(V, T))


def n_pow_4(n):
    return n ** 4

dataset = []
n = 0
T = 15
t_total = 100
VOLTS = np.arange(-90, 90, 10)
results = []




for V in VOLTS:
    voltage_results = []
    for t in np.arange(0, t_total):
        n = n_inf(V, T) * (1 - np.exp(-t/tau_n(V,T)))
        y = n_pow_4(n)
        voltage_results.append(y)
        dataset.append((t/1000, V, y))
    results.append(voltage_results)


graph_df = pd.DataFrame(results)
graph_df = graph_df.T
dataset_df = pd.DataFrame(dataset)

graph_df.plot()
plt.legend(np.arange(-90, 90, 10))
plt.xlabel("Time (ms)")
plt.show()

graph_df.to_csv('Prod/graph_df.csv', index=False)
dataset_df.to_csv('Prod/dataset.csv', index=False)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def alpha_n(V):
    return (0.01 * (V + 55)) / (1 - np.exp(-0.1 * (V + 55)))


def beta_n(V):
    return 0.125 * (np.exp(-0.0125 * ( V + 65 )))


def n_inf(V):
    return alpha_n(V) / (alpha_n(V) + beta_n(V))


def tau_n(V):
    return 1 / (alpha_n(V) + beta_n(V))


def n_pow_4(n):
    return n ** 4

dataset = []
n = 0
t_total = 100
VOLTS = np.arange(-90, 90, 10)
results = []




for V in VOLTS:
    voltage_results = []
    for t in np.arange(0, t_total):
        n = n_inf(V) * (1 - np.exp(-t/tau_n(V)))
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

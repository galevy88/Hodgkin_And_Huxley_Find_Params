import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def alpha_n(V, T):
    return 0.01 * (V + 55) / (1 - np.exp(-0.1 * (V + 55))) * np.exp(-0.0556 * (T - 25))


def beta_n(V, T):
    return 0.125 * np.exp(-0.0125 * (V + 65)) * np.exp(-0.0025 * (T - 25))


def n_inf(V, T):
    return alpha_n(V, T) / (alpha_n(V, T) + beta_n(V, T))


def tau_n(V, T):
    return 1 / (alpha_n(V, T) + beta_n(V, T))


def n_pow_4(n):
    return n ** 4


features_label_list = []
n = 0
T = 15
dt = 1
t_total = 100
VOLTS = np.arange(-90, 90, 10)
results = []


for V in VOLTS:
    print(V)
    voltage_results = []
    for t in np.arange(0, t_total, dt):
        n = n_inf(V, T) * (1 - np.exp(-t/tau_n(V,T)))
        y = n_pow_4(n)
        voltage_results.append(y)
        features_label_list.append((t/1000, V, y))
    results.append(voltage_results)


graph_df = pd.DataFrame(results)
graph_df = graph_df.T
features_label_df = pd.DataFrame(features_label_list)

graph_df.plot()
plt.legend(np.arange(-90, 90, 10))
plt.xlabel("Time (ms)")
plt.show()

graph_df.to_csv('Prod/graph_df.csv', index=False)
features_label_df.to_csv('Prod/features_label_df.csv', index=False)

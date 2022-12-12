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

def dn_dt(n, V, T, dt):
  return (n_inf(V, T) - n) / tau_n(V, T) * dt

def dV_dt(V, dt):
  #dV_dt = (V + 1 - V - 1) / (2 * dt)
  return 0

def n_pow_4(n):
  return n ** 4

gK = 36
gL = 0.3
EK = -88
EL = 54.4
n = 0
T = 37
Cm = 1
dt = 0.02
t_total = 100
VOLTS = np.arange(-90, 90, 10)

results = []

for V in VOLTS:
  print(V)
  voltage_results = []
  for t in np.arange(0, t_total, dt):
    dn = dn_dt(n, V, T, dt)
    n += dn
    Im = Cm * dV_dt(V, dt) + gL * (V - EL) + gK * n_pow_4(n) * (V - EK)
    voltage_results.append(Im)
  results.append(voltage_results)

df = pd.DataFrame(results)
df = df.T
df.loc[0] = [0] * len(df.columns)

df.plot()
plt.legend(np.arange(-90, 90, 10))
plt.xlabel("Time (ms)")
plt.ylabel("I (pA)")
plt.show()

df.to_csv('H_H_DF.csv')


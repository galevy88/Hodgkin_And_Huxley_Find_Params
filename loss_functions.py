import numpy as np
import HH_Equations as Equations
import random
from sklearn.preprocessing import StandardScaler
import pandas as pd

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

t, V, labels = get_data('Prod/dataset.csv')

def l2_loss(params, batchSize):
    loss = 0
    indices = random.sample(range(len(t)), batchSize)
    for i in indices:
        y_hat = np.round(Equations.get_y_hat(params, t[i], V[i]), 8)
        if y_hat == float("inf") or np.isnan(y_hat):
            y_hat = np.random.uniform(1,2) * 100
        loss += (np.round(y_hat,8) - np.round(labels[i],8)) ** 2
    if loss == float("inf") or np.isnan(loss): loss = np.random.uniform(1,2) * 1000
    ret_loss = loss / len(indices)
    return ret_loss


def l1_loss(params, batchSize):
    loss = 0
    indices = random.sample(range(len(t)), batchSize)
    for i in indices:
        y_hat = np.round(Equations.get_y_hat(params, t[i], V[i]), 8)
        if y_hat == float("inf") or np.isnan(y_hat):
            y_hat = np.random.uniform(1,2) * 100
        loss += np.abs(y_hat - labels[i])
    if loss == float("inf") or np.isnan(loss): loss = np.random.uniform(1,2) * 1000
    ret_loss = loss / len(indices)
    return ret_loss
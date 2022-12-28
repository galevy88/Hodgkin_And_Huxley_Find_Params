import torch
import pandas as pd
from functools import partial
from sklearn.preprocessing import StandardScaler

def alpha_n(V, c1, c2, c3 ,c4):
    result_alpha = (c1 * (V + c2)) / (1 - torch.exp(-c3 * (V + c4)))
    #print(f"result_alpha: {result_alpha}")
    return result_alpha

def beta_n(V, c5, c6, c7):
    result_beta = c5 * (torch.exp(-c6 * ( V + c7 )))
    #print(f"result_beta: {result_beta}")
    return result_beta

def n_inf(alpha, beta):
    result_n_inf =  alpha() / (alpha() + beta())
    #print(f"result   inf: {result_n_inf}")
    return result_n_inf

def tau_n(alpha, beta):
    result_tau = 1 / (alpha() + beta())
    #print(f"result tau  : {result_tau}")
    return result_tau

def n_pow_4(n):
    result_n =  n ** 4
    #print(f"result     n: {result_n}")
    return result_n


class Model(torch.nn.Module):
    def __init__(self, constants):
        super(Model, self).__init__()
        self.constants = torch.nn.Parameter(constants)

    def forward(self, inputs):
        t, V = inputs
        c1, c2, c3, c4, c5, c6, c7 = self.constants
        print(c1, c2, c3, c4, c5, c6, c7)
        alpha = partial(alpha_n, V=V, c1=c1, c2=c2, c3=c3, c4=c4)
        beta = partial(beta_n, V=V, c5=c5, c6=c6, c7=c7)
        n = n_inf(alpha, beta) * (1 - torch.exp(-t/tau_n(alpha, beta)))
        y = n_pow_4(n)
        return y

    def get_params(self):
        return self.constants

def loss_fn(model, inputs, labels):
    predictions = model(inputs)
    # print(f"Predict: {predictions}")
    # print(f"True: {labels}")
    return torch.nn.functional.l1_loss(predictions, labels)


def train_model(model, optimizer, nepochs, inputs, labels):
    for e in range(nepochs):
        running_loss = 0
        for i in range(len(inputs)):
            input = torch.tensor(inputs[i])
            label = torch.tensor(labels[i])
            #print(f"INPUT {input} LABEL {label}")
            optimizer.zero_grad()
            loss = loss_fn(model, input, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print("Epoch: {}/{}.. ".format(e+1, nepochs), "Training Loss: {:.3f} ".format(running_loss/len(inputs)))

#constants = torch.tensor([0.03,51,-0.3,45,0.185,-0.0225,55])
constants = torch.tensor(torch.randn(7))

model = Model(constants)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
nepochs = 5
data = pd.read_csv('Prod/dataset.csv')

data = data.values
scaler = StandardScaler()
data_standardized = scaler.fit_transform(data)
inputs = data_standardized[:, :-1]
labels = data[:, -1]
train_model(model, optimizer, nepochs, inputs, labels)
nums = model.get_params()
for number in nums:
    print('{:.5f}'.format(number))


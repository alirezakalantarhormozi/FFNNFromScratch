from Layer import Layer
from Neuron import Neuron
# t = Layer(1, False, False)
# # print(t)

# n = Neuron(5)
# print(n*5)

from Model import Model
import numpy as np
import pandas as pd


csv = pd.read_csv('lab2.csv')
# print(csv.iloc[:, [0, 1]])

train_x = csv.iloc[:25000, [0, 1]]
train_y = csv.iloc[:25000, [2, 3]]
# print(train_y)
print(type(train_y.to_numpy()))

m = Model(2, 2, [3], 100, 0.01, 0.9)

test_x = csv.iloc[25000:32000, [0, 1]]
test_y = csv.iloc[25000:32000, [2, 3]]

# m.fit(np.array([[0.1, 0.2]]), np.array([[0.5, 0]]))
m.fit(train_x.to_numpy(), train_y.to_numpy(), test_x.to_numpy(), test_y.to_numpy())

print(m)

# text_x = csv.iloc[10000:20000, [0, 1]]
# text_y = csv.iloc[10000:20000, [2, 3]]
# # print(text_x)
# print(text_y)
result = m.predict(test_x.to_numpy(), test_y.to_numpy())
print(result)

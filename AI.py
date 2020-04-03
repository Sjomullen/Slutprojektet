import numpy as np 
import matplotlib.pyplot as plt

#  length, width, type/color
data = [[3,    1.5,  1],
        [2,    1,    0],
        [4,    1.5,  1],
        [3,    1,    0],
        [3.5,  0.5,  1],
        [2,    0.5,  0],
        [5.5,  1,    1],
        [1,    1,    0]]

mystery_flower = [4.5, 1]

w1 = np.random.rand()
w2 = np.random.rand()
b = np.random.rand()

def sigmoid(x):
    return(1/1 + np.exp(-x))

T = np.linspace(-5, 5, 10)
Y = sigmoid(T)
Y

plt.plot(T, Y)

plt.show()
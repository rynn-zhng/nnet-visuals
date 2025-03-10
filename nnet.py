import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['figure.autolayout'] = True

class ANN:

    def __init__(self, inputs, outputs, epochs=50):
        self.M = inputs
        self.N = outputs
        self.epochs = range(epochs)

    def __call__(self, ax, inx, outy):
        x, y = np.array(inx), np.array(outy)
        bias = -1

        for epoch in self.epochs:
            for i in self.axis:
                if i == self.axis[0]:
                    self.layers[i] = x @ self.weights[i] + bias
                    self.slayers[i] = self.sigmoid(self.layers[i])
                else:
                    self.layers[i] = self.slayers[i+1] @ self.weights[i] + bias
                    self.slayers[i] = self.sigmoid(self.layers[i])

            dx = {}
            for i in self.raxis:
                if i == self.raxis[0]:
                    error = (y - self.slayers[i])**2
                    delta = 2.0*(y - self.slayers[i])*self.sigmoid(self.layers[i], derv=True)
                    dx[i] = delta
                else:
                    error = self.weights[i-1] @ delta
                    delta = error * self.sigmoid(self.layers[i], derv=True)
                    dx[i] = delta

            for i in self.axis:
                self.weights[i] -= dx[i]
            
        for p, i in enumerate(self.axis):
            z = self.weights[i]
            mm, nn = z.shape
            xx, yy = np.meshgrid(range(nn), range(mm))
            ax[p].cla()
            ax[p].set_title(f'Weights: {p+1}')
            ax[p].contourf(xx, yy, z, cmap='viridis')

        plt.pause(0.001)

    def sigmoid(self, x, derv=False):
        f = 1.0 / (1.0 + np.exp(-x))
        if derv:
            return f*(1 - f)
        return f


    def build(self):
        M, N = self.M, self.N 
        self.axis = list(range(M, N, -1))
        self.raxis = self.axis[::-1]

        self.weights = {}
        self.layers = {}
        self.slayers = {}

        for i in self.axis:
            self.weights[i] = 0.05*np.random.rand(i, i-1)
            self.layers[i] = np.zeros(i-1)
            self.slayers[i] = np.zeros(i-1)


fig = plt.figure(figsize=(10, 7))
plots = [fig.add_subplot(2, 3, i) for i in range(1, 7)]

ai = ANN(9, 3, epochs=100)
ai.build()

rows = 200
dataset = np.random.rand(rows, 9)
output = np.random.rand(rows, 3)

for index, (xx, yy) in enumerate(zip(dataset, output)):
    print("Number of rows left: ", rows - index)
    ai(plots, xx, yy)


plt.show()
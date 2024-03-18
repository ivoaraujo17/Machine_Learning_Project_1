import numpy as np
from matplotlib import pyplot as plt


class RegressaoLogistica:

    def __init__(self, X_, y_) -> None:
        self.y = np.array(y_)
        self.X = np.array(X_)
        # verifica se X_ tem a coluna 0 preenchida com 1
        if self.X[0][0] != 1:
            self.X = np.c_[np.ones((len(self.X), 1)), self.X]
        self.w = np.zeros((self.X.shape[1], 1))

    def sigmoid(self, z):
        sig = 1 / (1 + np.exp(-z))
        return sig
    
    def custo(self, theta):
        z = np.dot(self.X, theta)
        custo0 = np.dot(self.y.T, np.log(self.sigmoid(z) + 1e-15)) # Adicionar epsilon para evitar log(0)
        custo1 = np.dot( (1 - self.y).T , np.log(1 - self.sigmoid(z) + 1e-15)) # Adicionar epsilon para evitar log(0)
        custo = -((custo1 + custo0)) / len(self.y)
        return custo

    def fit(self, alpha=0.0001, iteracoes=400):
        lista_custos = np.zeros(iteracoes,)
        for i in range(iteracoes):
            self.w = self.w - alpha * np.dot(self.X.T, self.sigmoid(np.dot(self.X, self.w)) - np.reshape(self.y, (len(self.y), 1)))
            lista_custos[i] = self.custo(self.w).item()

    def predict(self, X_):
        z = np.dot(X_, self.w)
        lista = []
        for i in self.sigmoid(z):
            if i > 0.5:
                lista.append(1)
            else:
                lista.append(0)
        return lista

    def plot(self):
        X1 = self.X[self.y == 1]
        X2 = self.X[self.y == 0]
        plt.plot(X1[:, 1], X1[:, 2], 'ro')
        plt.plot(X2[:, 1], X2[:, 2], 'bo')
        plt.plot(self.X, (-self.w[0] - self.w[1]*self.X) / self.w[2], c='orange')
        # limita com o maior e menor valor de x e y
        plt.xlim(np.min(self.X[:, 1]) - 1, np.max(self.X[:, 1]) + 1)
        plt.ylim(np.min(self.X[:, 2]) - 1, np.max(self.X[:, 2]) + 1)      
        plt.show()


    
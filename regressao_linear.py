import numpy as np
from matplotlib import pyplot as plt

class RegressaoLinear:
    def __init__(self, X_, y_) -> None:
        self.y = np.array(y_)
        self.X = np.array(X_)
        # verifica se X_ tem a coluna 0 preenchida com 1
        if self.X[0][0] != 1:
            self.X = np.c_[np.ones((len(self.X), 1)), self.X]
        self.w = np.zeros((self.X.shape[1], 1))

    def fit(self):
        X_inversa = np.linalg.inv(np.dot(self.X.T , self.X))
        self.w = np.dot(X_inversa, np.dot(self.X.T, self.y))

    def predict(self, x):
        # se x é um vetor com tamanho de w - 1
        if len(x) == len(self.w) - 1:
            x = np.insert(x, 0, 1)
        # se x é uma matriz com n linhas e w - 1 colunas
        elif len(x[0]) == len(self.w) - 1:
            # adicona a coluna de 1
            x = np.c_[np.ones((len(x), 1)), x]

        y_predito = np.dot(x, self.w)
        
        return np.array([1 if y > 0 else -1 for y in y_predito])
    
    def plot(self):
        X1 = self.X[self.y == 1]
        X2 = self.X[self.y == -1]
        plt.plot(X1[:, 1], X1[:, 2], 'ro')
        plt.plot(X2[:, 1], X2[:, 2], 'bo')
        plt.plot(self.X, (-self.w[0] - self.w[1]*self.X) / self.w[2], c='orange')
        # limita com o maior e menor valor de x e y
        plt.xlim(np.min(self.X[:, 1]) - 0.5, np.max(self.X[:, 1]) + 0.5)
        plt.ylim(np.min(self.X[:, 2]) - 0.5, np.max(self.X[:, 2]) + 0.5)       
        plt.show()
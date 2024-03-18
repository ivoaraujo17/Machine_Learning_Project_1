import numpy as np
from matplotlib import pyplot as plt


class PlaPocket():
    def __init__(self, X_, y_) -> None:
        self.y = np.array(y_)
        self.X = np.array(X_)
        # verifica se X_ tem a coluna 0 preenchida com 1
        if self.X[0][0] != 1:
            self.X = np.c_[np.ones((len(self.X), 1)), self.X]
        self.w = np.zeros(len(self.X[0]))
        self.iteracao = 0

    def fit(self):
        erro_max = self.y.size
        melhor_w = self.w
        while self.iteracao < 100000 and erro_max > 0:
            self.iteracao += 1
            erro_atual = self.__erro_amostra()
            
            if erro_atual < erro_max:
                erro_max = erro_atual
                melhor_w = self.w
            for i in range(self.y.size):
                if np.sign(np.dot(self.w, self.X[i])) != self.y[i]:
                    self.w = self.w + np.dot(self.X[i], self.y[i])

        self.w = melhor_w

    def get_w(self):
        return self.w

    def predict(self, x):
        return np.sign(np.dot(self.w, x))
    
    def __erro_amostra(self):
        error = 0
        for i in range(self.y.size):
            if np.sign(np.dot(self.w, self.X[i])) != self.y[i]:
                error += 1   
        return error

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

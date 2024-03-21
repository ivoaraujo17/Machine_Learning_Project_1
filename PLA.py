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
        # se x é um vetor com tamanho de w - 1
        if len(x) == len(self.w) - 1:
            x = np.insert(x, 0, 1)
        # se x é uma matriz com n linhas e w - 1 colunas
        elif len(x[0]) == len(self.w) - 1:
            # adicona a coluna de 1
            x = np.c_[np.ones((len(x), 1)), x]

        return np.sign(np.dot(x, self.w))
    
    def __erro_amostra(self):
        error = 0
        for i in range(self.y.size):
            if np.sign(np.dot(self.w, self.X[i])) != self.y[i]:
                error += 1   
        return error

    def plot(self):
        X1 = self.X[self.y == 1]
        X2 = self.X[self.y == -1]
        plt.scatter(X1[:, 1], X1[:, 2], c='blue', label='1')
        plt.scatter(X2[:, 1], X2[:, 2], c='red', label='-1')
        plt.plot(self.X, (-self.w[0] - self.w[1]*self.X) / self.w[2], c='orange')
        plt.legend()
        # limita com o maior e menor valor de x e y
        plt.xlim(np.min(self.X[:, 1]) - 0.5, np.max(self.X[:, 1]) + 0.5)
        plt.ylim(np.min(self.X[:, 2]) - 0.5, np.max(self.X[:, 2]) + 0.5)   
        plt.show()    

import numpy as np
from matplotlib import pyplot as plt

class PlaPocket():
    def execute(self, X_, y):
        X = np.array(X_)
        self.w = np.zeros(len(X_[0]))
        erro_max = len(y)
        melhor_w = self.w

        iteracao = 0
        while iteracao < 100000 and erro_max > 0:
            iteracao += 1
            erro_atual = self.erro_amostra(X,y)
            if erro_atual < erro_max:
                erro_max = erro_atual
                melhor_w = self.w
            for i in range(len(y)):
                if np.sign(np.dot(self.w, X[i])) != y[i]:
                    self.w += np.dot(X[i], y[i])
                    
        print(iteracao)
        self.w = melhor_w
    
    def hipotese_w(self, x):
        return np.sign(np.dot(self.w, x))
    
    def erro_amostra(self, X, y):
        error = 0
        for i in range(len(y)):
            if np.sign(np.dot(self.w, X[i])) != y[i]:
                error += 1   
        return error 
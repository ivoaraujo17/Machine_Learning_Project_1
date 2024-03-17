import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

dados = pd.read_csv("Dados/train_reduzido_filter_1_5.csv", sep=";")
X_treino = dados[['simetria', 'intensidade']].values
y_treino = dados['label'].values
dados = pd.read_csv("Dados/test_reduzido_filter_1_5.csv", sep=";")
X_teste = dados[['simetria', 'intensidade']].values
y_teste = dados['label'].values
y_teste[y_teste == 1] = 0
y_teste[y_teste == 5] = 1
y_treino[y_treino == 1] = 0
y_treino[y_treino == 5] = 1

def pontuacao_F1(y, y_hat):
    tp, tn, fp, fn = 0, 0, 0, 0

    for i in range(len(y)):
        if y[i] == 1 and y_hat[i] == 1:
            tp += 1
        elif y[i] == 1 and y_hat[i] == 0:
            fn += 1
        elif y[i] == 0 and y_hat[i] == 1:
            fp += 1
        elif y[i] == 0 and y_hat[i] == 0:
            tn += 1
    precisao = tp / (tp + fp)
    recall = tp / (tp + fn)
    pontuacao_f1 = 2 * precisao * recall / (precisao + recall)
    return pontuacao_f1

def padronizar(X_tr):
    for i in range(X_tr.shape[1]):
        X_tr[:, i] = (X_tr[:, i] - np.mean(X_tr[:, i])) / np.std(X_tr[:, i])
    return X_tr

X_treino = padronizar(X_treino)
X_teste = padronizar(X_teste)

class RegressaoLinear:
    def __init__(self):
        self.pesos = None

    def inicializar(self, X):
        pesos = np.zeros((X.shape[1] + 1, 1))
        X = np.c_[np.ones((X.shape[0], 1)), X]
        return pesos, X

    def ajustar(self, X, y):
        pesos, X = self.inicializar(X)
        pesos = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        self.pesos = pesos
        return pesos

    def prever(self, X):
        X = np.c_[np.ones((X.shape[0], 1)), X]
        y_predito = X.dot(self.pesos)
        for i in range(len(y_predito)):
            if y_predito[i] > 0.5:
                y_predito[i] = 1
            else:
                y_predito[i] = 0
        return y_predito

obj2 = RegressaoLinear()
pesos = obj2.ajustar(X_treino, y_treino)
y_predito = obj2.prever(X_teste)
y_treino_predito = obj2.prever(X_treino)
print(pesos)
# Vamos ver a pontuação F1 para os dados de treinamento e teste
pontuacao_f1_tr = pontuacao_F1(y_treino, y_treino_predito)
pontuacao_f1_te = pontuacao_F1(y_teste, y_predito)

print("Testando com a amostra que foi treinada:", pontuacao_f1_tr)
print("Testando com a amostra de fora:", pontuacao_f1_te)
 # Plotando o limite de decisão
x1 = np.linspace(np.min(X_treino[:, 0]), np.max(X_treino[:, 0]), 100)
x2 = np.linspace(np.min(X_treino[:, 1]), np.max(X_treino[:, 1]), 100)
xx1, xx2 = np.meshgrid(x1, x2)
X_grid = np.c_[xx1.ravel(), xx2.ravel()]
y_grid = obj2.prever(X_grid)
y_grid = np.array(y_grid).reshape(xx1.shape)

plt.contourf(xx1, xx2, y_grid, alpha=0.1)
plt.scatter(X_teste[:, 0], X_teste[:, 1], c=y_predito)
plt.xlabel('Simetria')
plt.ylabel('Intensidade')
plt.title('Regressão Linear - Limite de Decisão')
plt.show()
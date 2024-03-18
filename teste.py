import numpy as np 
from numpy import log,dot,exp,shape
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

def padronizar(X_tr):
    for i in range(shape(X_tr)[1]):
        X_tr[:,i] = (X_tr[:,i] - np.mean(X_tr[:,i]))/np.std(X_tr[:,i])
        #X_tr[:,i] = np.sign(X_tr[:,i])

    return X_tr
X_treino = padronizar(X_treino)
X_teste = padronizar(X_teste)

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
class RegressaoLogistica:
    def sigmoid(self, z):
        sig = 1 / (1 + exp(-z))
        return sig
    def inicializar(self, X):
        pesos = np.zeros((shape(X)[1] + 1, 1))
        X = np.c_[np.ones((shape(X)[0], 1)), X]
        return pesos, X
    def ajustar(self, X, y, alpha=0.0001, iteracoes=400):
        pesos, X = self.inicializar(X)
        def custo(theta):
            z = dot(X, theta)
            custo0 = y.T.dot(log(self.sigmoid(z) + 1e-15))  # Adicionar epsilon para evitar log(0)
            custo1 = (1 - y).T.dot(log(1 - self.sigmoid(z) + 1e-15))  # Adicionar epsilon para evitar log(0)
            custo = -((custo1 + custo0)) / len(y)
            return custo
        lista_custos = np.zeros(iteracoes,)
        for i in range(iteracoes):
            pesos = pesos - alpha * dot(X.T, self.sigmoid(dot(X, pesos)) - np.reshape(y, (len(y), 1)))
            lista_custos[i] = custo(pesos).item()
        self.pesos = pesos
        return lista_custos, pesos
    def prever(self, X):
        z = dot(self.inicializar(X)[1], self.pesos)
        lista = []
        for i in self.sigmoid(z):
            if i > 0.5:
                lista.append(1)
            else:
                lista.append(0)
        return lista

obj1 = RegressaoLogistica()
modelo, pesos = obj1.ajustar(X_treino, y_treino)
y_predito = obj1.prever(X_teste)
y_treino = obj1.prever(X_treino)

# Vamos ver a pontuação F1 para os dados de treinamento e teste
pontuacao_f1_tr = pontuacao_F1(y_treino, y_treino)
pontuacao_f1_te = pontuacao_F1(y_teste, y_predito)

print("Testando com a amostra que foi treinada:", pontuacao_f1_tr)
print("Testando com a amostra de fora:", pontuacao_f1_te)
# Plotando o limite de decisão
X1 = X_teste[y_teste == 1]
X2 = X_teste[y_teste == 0]
plt.plot(X1[:, 0], X1[:, 1], 'ro')
plt.plot(X2[:, 0], X2[:, 1], 'bo')
plt.plot(X_teste, (-pesos[0] - pesos[1]*X_teste) / pesos[2], c='orange')
plt.show()

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns


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
        while self.iteracao < 10000 and erro_max > 0:
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
    
    def reverse_predict(self, y, lb_1, lb_2):
        return np.where(y == 1, lb_1, lb_2)
    
    def acuracia(self, y_test, y_predict_reversed, labels):
        print(f"\nAcurácia PLA")
        print(classification_report(y_test, y_predict_reversed, target_names=labels))
        # Matriz de confusão
        cm = confusion_matrix(y_test, y_predict_reversed)
        plt.figure(figsize=(5, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.xlabel("Classe Predita")
        plt.ylabel("Classe Real")
        plt.title("Matriz de Confusão")
        plt.show()

    def plot(self):
        X1 = self.X[self.y == 1]
        X2 = self.X[self.y == -1]
        plt.title(f"PLA - Iterações: {self.iteracao}")
        plt.xlabel("Intensidade")
        plt.ylabel("Simetria")
        plt.scatter(X1[:, 1], X1[:, 2], c='blue', label='1')
        plt.scatter(X2[:, 1], X2[:, 2], c='red', label='-1')
        xmin = np.min(self.X[:, 1]) - 0.5
        xmax = np.max(self.X[:, 1]) + 0.5
        x = np.linspace(xmin, xmax, 100)
        plt.plot(x, (-self.w[0] - self.w[1]*x) / self.w[2], label="w", c='orange')
        plt.legend()
        # limita com o maior e menor valor de x e y
        plt.xlim(xmin, xmax)
        plt.ylim(np.min(self.X[:, 2]) - 0.5, np.max(self.X[:, 2]) + 0.5)   
        plt.show()    

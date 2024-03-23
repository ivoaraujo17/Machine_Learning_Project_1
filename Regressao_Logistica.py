import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns


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
            # Calculate the predicted values
            predicted_values = self.sigmoid(np.dot(self.X, self.w))
            
            # Calculate the error
            error = predicted_values - np.reshape(self.y, (len(self.y), 1))
            
            # Update the weights
            self.w = self.w - alpha * np.dot(self.X.T, error)
            lista_custos[i] = self.custo(self.w).item()

    def predict(self, x):
        # se x é um vetor com tamanho de w - 1
        if len(x) == len(self.w) - 1:
            x = np.insert(x, 0, 1)
        # se x é uma matriz com n linhas e w - 1 colunas
        elif len(x[0]) == len(self.w) - 1:
            # adicona a coluna de 1
            x = np.c_[np.ones((len(x), 1)), x]

        z = np.dot(x, self.w)
        return np.array([1 if y > 0.5 else 0 for y in self.sigmoid(z)])
    
    def reverse_predict(self, y, lb_1, lb_2):
        return np.where(y == 1, lb_1, lb_2)
    
    def acuracia(self, y_test, y_predict_reversed, labels):
        print(f"\nAcurácia Regressão Logística")
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
        X2 = self.X[self.y == 0]
        plt.title(f"Regressão Logística")
        plt.xlabel("Intensidade")
        plt.ylabel("Simetria")
        plt.scatter(X1[:, 1], X1[:, 2], c='blue', label='1')
        plt.scatter(X2[:, 1], X2[:, 2], c='red', label='0')
        xmin = np.min(self.X[:, 1]) - 0.5
        xmax = np.max(self.X[:, 1]) + 0.5
        x = np.linspace(xmin, xmax, 100)
        plt.plot(x, (-self.w[0] - self.w[1]*x) / self.w[2], label="w", c='orange')
        plt.legend()
        # limita com o maior e menor valor de x e y
        plt.xlim(xmin, xmax)
        plt.ylim(np.min(self.X[:, 2]) - 0.5, np.max(self.X[:, 2]) + 0.5)       
        plt.show()


    
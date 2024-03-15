import numpy as np 
from numpy import log,dot,exp,shape
import pandas as pd
import matplotlib.pyplot as plt

dados = pd.read_csv("train_reduzido_filter_1_5.csv", sep=";")
X_train = dados[['simetria', 'intensidade']].values
y_train = dados['label'].values
dados = pd.read_csv("test_reduzido_filter_1_5.csv", sep=";")
X_test = dados[['simetria', 'intensidade']].values
y_test = dados['label'].values
y_test[y_test == 1] = 0
y_test[y_test == 5] = 1

def standardize(X_tr):
    for i in range(shape(X_tr)[1]):
        X_tr[:,i] = (X_tr[:,i] - np.mean(X_tr[:,i]))/np.std(X_tr[:,i])
        #X_tr[:,i] = np.sign(X_tr[:,i])

    return X_tr
X_train = standardize(X_train)
print(X_train)
X_test = standardize(X_test)

def F1_score(y, y_hat):
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
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)
    return f1_score
class LogisticRegression:
    def sigmoid(self, z):
        sig = 1 / (1 + exp(-z))
        return sig
    def initialize(self, X):
        weights = np.zeros((shape(X)[1] + 1, 1))
        X = np.c_[np.ones((shape(X)[0], 1)), X]
        return weights, X
    def fit(self, X, y, alpha=0.001, iter=400):
        weights, X = self.initialize(X)
        def cost(theta):
            z = dot(X, theta)
            cost0 = y.T.dot(log(self.sigmoid(z) + 1e-15))  # Add epsilon to avoid log(0)
            cost1 = (1 - y).T.dot(log(1 - self.sigmoid(z) + 1e-15))  # Add epsilon to avoid log(0)
            cost = -((cost1 + cost0)) / len(y)
            return cost
        cost_list = np.zeros(iter,)
        for i in range(iter):
            weights = weights - alpha * dot(X.T, self.sigmoid(dot(X, weights)) - np.reshape(y, (len(y), 1)))
            cost_list[i] = cost(weights).item()
        self.weights = weights
        return cost_list, weights
    def predict(self, X):
        z = dot(self.initialize(X)[1], self.weights)
        lis = []
        for i in self.sigmoid(z):
            if i > 0.5:
                lis.append(1)
            else:
                lis.append(0)
        return lis

obj1 = LogisticRegression()
model, weights = obj1.fit(X_train, y_train)
y_pred = obj1.predict(X_test)
y_train = obj1.predict(X_train)

# Let's see the f1-score for training and testing data
f1_score_tr = F1_score(y_train, y_train)
f1_score_te = F1_score(y_test, y_pred)

print("testando com a amostra que foi treinada:",f1_score_tr)
print("testando com a amostra de fora:",f1_score_te)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
plt.xlabel('Simetria')
plt.ylabel('Intensidade')
plt.title('Logistic Regression - Decision Boundary')
plt.show()


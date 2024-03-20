import numpy as np

def padronizar(X_tr):
    for i in range(np.shape(X_tr)[1]):
        X_tr[:,i] = (X_tr[:,i] - np.mean(X_tr[:,i]))/np.std(X_tr[:,i])
        
    return X_tr

def acuracia(y, y_pred):
    return (np.sum(y == y_pred)/len(y))*100

def reverse_predict(y, lb_1, lb_2):
    if y == 1:
        return lb_1
    else:
        return lb_2

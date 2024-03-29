import numpy as np


def padronizar(X_tr, media_std=None):
    # cria um vetor de zeros com a quantidade de colunas de X_tr e duas linhas (média e desvio padrão)
    paramentros = np.zeros((2, np.shape(X_tr)[1]))
    
    for i in range(np.shape(X_tr)[1]):
        if media_std is None:
            # calcula a média
            paramentros[0, i] = np.mean(X_tr[:,i])
            # calcula o desvio padrão
            paramentros[1, i] = np.std(X_tr[:,i])
        else:
            paramentros[0, i] = media_std[0, i]
            paramentros[1, i] = media_std[1, i]
        # padroniza os dados
        X_tr[:,i] = (X_tr[:,i] - paramentros[0, i])/paramentros[1, i]

    return X_tr, paramentros


def filter(path, digito_1, digito_2):
    with open(path, "r") as arq:
        linhas = arq.readlines()
        nome = path.split(".")[0]
        with open(f"{nome}_filter_{digito_1}_{digito_2}.csv", "w") as arq_filter:
            arq_filter.write("label;intensidade;simetria\n")
            
            for linha in linhas[1:]:
                if float(linha.split(";")[0]) in [float(digito_1), float(digito_2)]:
                    arq_filter.write(linha)


def read(path):
    with open(path, "r") as arq:
        linhas = arq.readlines()
        nome = path.split(".")[0]
        with open(f"{nome}_reduzido.csv", "w") as arq_reduzido:
            arq_reduzido.write("label;intensidade;simetria\n")
            for linha in linhas[1:]:
                linha_ = linha.split(";")
                y = float(linha_[0])
                X = np.zeros((28, 28))
                linha = 0
                coluna = 0
                for x in linha_[1:]:
                    X[linha, coluna] = float(x)
                    if coluna == 27:
                        linha += 1
                        coluna = 0
                    else:
                        coluna += 1
                # calcula intensidade
                intensidade = X.sum()/255
                # calcula simetria
                simetria_v = 0
                for i in range(28):
                    for j in range(14):
                        simetria_v += abs(X[i, j] - X[i, 27-j])

                simetria_v = simetria_v/255

                simetria_h = 0
                for j in range(28):
                    for i in range(14):
                        simetria_h += abs(X[i, j] - X[27-i, j])
                
                simetria_h = simetria_h/255

                simetria = simetria_v + simetria_h
                arq_reduzido.write(f"{y};{intensidade};{simetria}\n") 

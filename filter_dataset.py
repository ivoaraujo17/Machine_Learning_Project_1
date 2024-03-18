
def filter(path, digito_1, digito_2):
    with open(path, "r") as arq:
        linhas = arq.readlines()
        nome = path.split(".")[0]
        with open(f"{nome}_filter_{digito_1}_{digito_2}.csv", "w") as arq_filter:
            arq_filter.write("label;intensidade;simetria\n")
            
            for linha in linhas[1:]:
                if float(linha.split(";")[0]) in [float(digito_1), float(digito_2)]:
                    arq_filter.write(linha)

if __name__ == "__main__":
    filter("Dados/test_reduzido.csv", 4, 5)
    filter("Dados/train_reduzido.csv", 4, 5)
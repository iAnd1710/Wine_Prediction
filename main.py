#imports
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

#Carregando o conjunto de dados 
arquivo = pd.read_csv(' caminho do csv ')

#Transformando coluna Style em numérico para poder fazer cálculos
arquivo['style'] = arquivo['style'].replace('red', 0)
arquivo['style'] = arquivo['style'].replace('white', 1)

#Separando as variáveis entre preditoras e variável alvo
y = arquivo['style']
x = arquivo.drop('style', axis=1)

#Criando os conjuntos de treino e teste:
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size = 0.3)

#Criando o modelo:
modelo = ExtraTreesClassifier()
modelo.fit(x_treino,y_treino)

#Imprimindo resultados
resultado = modelo.score(x_teste, y_teste)
print("Acurácia: ", resultado)
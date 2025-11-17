import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression       

# lendo o arquivo
df = pd.read_csv("consumo_industrial.csv")

# separando as colunas
X = df["horas_trabalhadas"].values.reshape(-1, 1)
y = df["consumo_kwh"].values

# criando o modelo de regressão linear
model = LinearRegression()
model.fit(X, y)

# pegando os valores de w e b
w = model.coef_[0]
b = model.intercept_

print(f"Equação da reta: y = {w:.2f}x + {b:.2f}")

# previsões
y_pred = model.predict(X)

# gráfico com dispersão + linha
plt.scatter(X, y, label="Dados Reais")
plt.plot(X, y_pred, color="red", label="Linha de Regressão")
plt.xlabel("Horas Trabalhadas")
plt.ylabel("Consumo (kWh)")
plt.title("Consumo Industrial vs Horas Trabalhadas")
plt.legend()
plt.grid()
plt.show()

# classificação de alto consumo
media = df["consumo_kwh"].mean()
desvio = df["consumo_kwh"].std()
limite = media + desvio

df["alto_consumo"] = df["consumo_kwh"] > limite

print("\nTabela final:")
print(df)

print("\nDias de alto consumo:")
print(df[df["alto_consumo"] == True])

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

print("funcionando")

# Setup
data = pd.read_csv("pokemon.csv")
data = data.query("Type1.isin(('Grass', 'Electric'))")
X = data[["Attack", "Speed", "Sp. Atk", "Sp. Def", "Total"]]
Y = (data["Type1"] == "Grass")

# Split
# ADICIONADO random_state=42 para garantir que a divisão seja sempre a mesma
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Listas para guardar os resultados
listaf1 = []
listamaxdx =[]

# Loop for (mais limpo que o while para este caso)
for max_depth in range(2, 101): # Vai do 2 até o 100
    # ADICIONADO random_state=42 na Árvore de Decisão
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=20).fit(X_train, Y_train)
    
    # Predict
    answer = tree.predict(X_test)
    
    # Calcula F1
    f1 = f1_score(Y_test, answer)
    
    # Adiciona nas listas usando o método .append()
    listaf1.append(f1)
    listamaxdx.append(max_depth)
    
    print(f"Max Depth: {max_depth} | F1 Test Score: {f1:.4f}")

# Plotagem do Gráfico
plt.figure(figsize=(10, 6)) # Define um tamanho legal para o gráfico
plt.plot(listamaxdx, listaf1, marker='o', linestyle='-') # marker='o' coloca bolinhas nos pontos

# Adicionando detalhes para o gráfico ficar profissional
plt.title("F1 Score vs Max Depth")
plt.xlabel("Max Depth")
plt.ylabel("F1 Score")
plt.xticks(range(2, 21)) # Garante que todos os números do eixo X apareçam
plt.grid(True) # Adiciona uma grade no fundo para facilitar a leitura

plt.show()
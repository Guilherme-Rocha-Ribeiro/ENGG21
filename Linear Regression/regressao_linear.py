import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "Regressão(Exemplo Preço).xlsx")

df = pd.read_excel(file_path)


X = df[['Carne (g)', 'Batata (g)', 'Cerveja (um)']].to_numpy(dtype=float) # create matrix with 3 columns. where each row is the value 0j, 1j, 2j from its respective 
# Converter preços para float
df['Preço (R$)'] = df['Preço (R$)'].str.replace(',', '.').astype(float) # changes ',' to '.' on the string
y = df['Preço (R$)'].to_numpy(dtype=float) 


# params = np.linalg.inv(X.T @ X) @ (X.T @ y) # @ -> matrix multiplication notation
# print(f"  p1 (Preço por g Carne): {params[0]:.4f}")
# print(f"  p2 (Preço por g Batata): {params[1]:.4f}")
# print(f"  p3 (Preço por uni Cerveja): {params[2]:.4f}\n")





params = np.linalg.inv(X.T @ X) @ (X.T @ y)
print("Manual implementation results:")
print(f"  p1 (Preço por g Carne): {params[0]}")
print(f"  p2 (Preço por g Batata): {params[1]}")
print(f"  p3 (Preço por uni Cerveja): {params[2]}\n")
y_modelo = X @ params
df['modelo'] = y_modelo
print(df[['Preço (R$)', 'modelo']])

# Using scikit-learn LinearRegression for validation
model = LinearRegression(fit_intercept=False)  # Set to False since you're not including intercept manually
model.fit(X, y)

print("Scikit-learn validation results:")
print(f"  p1 (Preço por g Carne): {model.coef_[0]}")
print(f"  p2 (Preço por g Batata): {model.coef_[1]}")
print(f"  p3 (Preço por uni Cerveja): {model.coef_[2]}")

# Compare coefficients
print("\nCoefficient comparison:")
print(f"  Carne difference: {abs(params[0] - model.coef_[0])}")
print(f"  Batata difference: {abs(params[1] - model.coef_[1])}")  
print(f"  Cerveja difference: {abs(params[2] - model.coef_[2])}")








fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Análise do Modelo de Regressão', fontsize=16)

# Gráfico 1: Carne vs Preço
axes[0,0].scatter(X[:,0], y, alpha=0.7, label='Dados')
axes[0,0].scatter(X[:,0], y_modelo, alpha=0.7, color='red', label='Modelo')
axes[0,0].set_xlabel('Carne (g)')
axes[0,0].set_ylabel('Preço (R$)')
axes[0,0].set_title('Carne vs Preço')
axes[0,0].legend()
axes[0,0].grid(True)

# Gráfico 2: Batata vs Preço
axes[0,1].scatter(X[:,1], y, alpha=0.7, label='Dados')
axes[0,1].scatter(X[:,1], y_modelo, alpha=0.7, color='red', label='Modelo')
axes[0,1].set_xlabel('Batata (g)')
axes[0,1].set_ylabel('Preço (R$)')
axes[0,1].set_title('Batata vs Preço')
axes[0,1].legend()
axes[0,1].grid(True)

# Gráfico 3: Cerveja vs Preço
axes[1,0].scatter(X[:,2], y, alpha=0.7, label='Dados')
axes[1,0].scatter(X[:,2], y_modelo, alpha=0.7, color='red', label='Modelo')
axes[1,0].set_xlabel('Cerveja (un)')
axes[1,0].set_ylabel('Preço (R$)')
axes[1,0].set_title('Cerveja vs Preço')
axes[1,0].legend()
axes[1,0].grid(True)

# Gráfico 4: Preço Real vs Predito
axes[1,1].scatter(y, y_modelo, alpha=0.7)
axes[1,1].set_xlabel('Preço Real (R$)')
axes[1,1].set_ylabel('Preço Predito (R$)')
axes[1,1].set_title('Real vs Predito')
axes[1,1].grid(True)

plt.tight_layout()
plt.show()


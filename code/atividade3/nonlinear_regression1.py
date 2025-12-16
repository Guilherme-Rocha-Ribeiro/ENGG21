import kagglehub
# https://www.kaggle.com/datasets/mirajdeepbhandari/polynomial-regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split 


plt.style.use(['seaborn-v0_8-paper'])

path = kagglehub.dataset_download("mirajdeepbhandari/polynomial-regression")
df = pd.read_csv(path + '/Ice_cream selling data.csv')

target = 'Ice Cream Sales (units)'
features = 'Temperature (°C)'

u = df[features]
y = df[target]

# utd = u_training_data
# uvd = u_validation_data 
# ytd = y_training_data
# yvd = y_validation_data
utd, uvd, ytd, yvd = train_test_split(u,y, test_size=0.4)

# nn = neurons_number
best_loss = 1000
for nn in range(1, 15):
    rna = keras.models.Sequential()
    n_cols = 1
    # Input layer
    rna.add(keras.layers.Input(shape=(n_cols,)))
    
    # Hidden Layers
    # elu because the input has negative values that have to be considered
    rna.add(keras.layers.Dense(nn, activation='elu')) # min_error with one layer = 16.19
    rna.add(keras.layers.Dense(4, activation='elu'))  # min_error with 2 layers = 8.89
    rna.add(keras.layers.Dense(2, activation='relu'))  # min_error with 3 layers = 6.94
    
    
    # Output layer
    rna.add(keras.layers.Dense(1, activation='linear')) 
    
    # Compile
    opt = keras.optimizers.Adam(learning_rate=0.05)
    rna.compile(optimizer=opt, 
                loss='mse',    # Mean Squared Error
                metrics=['mae']) # Mean Absolute Error | Root MSE
    
    training = rna.fit(utd, ytd,
                      epochs=800, 
                      verbose=0,
                      validation_data=(uvd, yvd))
    rna.summary()
    current_val_loss = training.history['val_loss'][-1]
    if current_val_loss < best_loss:
        print(f"Novo melhor modelo! Neurônios: {nn}, Loss Validação: {current_val_loss:.2f}, loss: {training.history['loss'][-1]}")
        best_loss = current_val_loss
        best_rna = rna
    
    
plt.plot(training.history['loss'])
plt.xlabel('Épocas')
plt.ylabel('Erro')
plt.show()

print("---")
print(f"O melhor modelo tem {best_rna.layers[0].units} neurônios.")

# --- Como avaliar um modelo de regressão ---

# 1. Fazer predições nos dados de validação
y_pred = best_rna.predict(uvd)
figure, axes = plt.subplots(1, 2, figsize=(12, 10))  # 1 linha, 2 colunas

# Primeiro gráfico (lado esquerdo)
ax = axes[0]
ax.scatter(uvd, yvd, color='blue', label='Dados Reais (Validação)') 
ax.scatter(uvd, y_pred, color='black', label='Predições do Modelo')
ax.set_title('Comparação: Dados vs Predições')
ax.set_xlabel('Temperatura (°C)')
ax.set_ylabel('Vendas de Sorvete (unidades)')
ax.legend()
ax.grid(ls='--', lw=0.5)

# Segundo gráfico (lado direito)
ax = axes[1]
x_continuo = np.linspace(u.min(), u.max(), 100) 
y_funcao = best_rna.predict(x_continuo.reshape(-1, 1))
ax.scatter(uvd, yvd, color='blue', label='Dados Reais (Validação)')
ax.plot(x_continuo, y_funcao, color='red', label='Função do Modelo (Predições)', linewidth=2)
ax.set_title('Função de Regressão do Modelo')
ax.set_xlabel('Temperatura (°C)')
ax.set_ylabel('Vendas de Sorvete (unidades)')
ax.legend()
ax.grid(ls='--', lw=0.5)

plt.suptitle('Performance do Modelo de Regressão', fontsize=16)
plt.show()
    
pass
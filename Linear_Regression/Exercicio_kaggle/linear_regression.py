import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from timeit import timeit

# The row data_train[x = 213][y = 213] had the pair (3530, NaN). It was needed to clean the data before continue the training

path = kagglehub.dataset_download("andonians/random-linear-regression")
print("Path to dataset files:", path)

data_train =  pd.read_csv(f"{path}/train.csv")
# Remove any row with NaN in either column pd.dropna
train_valids = data_train.dropna(subset=['x', 'y'])
x_train = train_valids['x'].to_numpy(dtype=float).reshape(-1, 1)
y_train = train_valids['y'].to_numpy(dtype=float).reshape(-1, 1)

# time_full_npdot = timeit(lambda: np.dot(np.linalg.inv(np.dot(x_train.T, x_train)), np.dot(x_train.T, y_train)) , number=10000)
# print(f"full npdot = {time_full_npdot}")

# time_partial_npdot = timeit(lambda: np.linalg.inv(np.dot(x_train.T, x_train)) @ (np.dot(x_train.T, y_train)), number=10000)
# print(f"partial npdot = {time_partial_npdot}")

# time_at = timeit(lambda: np.linalg.inv(x_train.T @ x_train) @ (x_train.T @ y_train), number=10000)
# print(f"full @ = {time_at}") 

# full np.dot >> partial npdot >> full @ 

# ============
# Executes the lambda function 10,000 times
# Measures the total time for all 10,000 executions
# Returns the total time (not the average per execution)
# ===========

train_params = np.dot(np.linalg.inv(np.dot(x_train.T, x_train)), np.dot(x_train.T, y_train))
print(f"TP = {train_params}")

model_ytrain = np.dot(x_train, train_params) 


TRAIN_sklear_model = LinearRegression(fit_intercept=False)  # Set to False since you're not including intercept manually
TRAIN_sklear_model.fit(x_train, y_train)
print(f"LR = {TRAIN_sklear_model.coef_}")
print(f" Validação com LinearRegression: {abs(abs(train_params - TRAIN_sklear_model.coef_))}")


# fig, axes = plt.subplots(2, 1, figsize=(12, 10))
# fig.suptitle('Análise do Modelo de Regressão', fontsize=16)

# axes[0].scatter(x_train, y_train, alpha=0.4, label='Dados')
# axes[0].plot(x_train, ytrain_model, alpha=0.9, color='red', label='Modelo')
# axes[0].set_xlabel('x')
# axes[0].set_ylabel('y')
# axes[0].legend()
# axes[0].grid(True)


# axes[1].scatter(y_train, ytrain_model, label='Modelo')
# axes[1].set_xlabel('y_dados')
# axes[1].set_ylabel('y_modelo')
# axes[1].set_title('y_dados vs y_modelo')
# axes[1].legend()
# axes[1].grid(True)

# plt.figure()                  # Abre a janela de gráfico
# plt.plot(y_train,ytrain_model,'.') # Dados experimental na forma de ponto
# plt.xlabel('y_experimentais') # Texto para o eixo x
# plt.ylabel('y_modelo')        # Texto para o eixo y
# plt.grid()                    # Mostra a grade
# plt.legend()                  # Legenda do gráfico
# plt.show()




data_test = pd.read_csv(f"{path}/test.csv")
test_valids = data_test.dropna(subset=['x', 'y'])
x_test = test_valids['x'].to_numpy(dtype=float).reshape(-1, 1) # how to use this data ?
y_test = test_valids['y'].to_numpy(dtype=float).reshape(-1, 1)








model_ytest = np.dot(y_test, train_params)
TEST_sklear_model = LinearRegression(fit_intercept=False)
TEST_sklear_model.fit(x_test, y_test)
print(TEST_sklear_model.coef_)


print(f" Validação com LinearRegression: {abs(abs(train_params - TEST_sklear_model.coef_))}")



# plt.figure()                  # Abre a janela de gráfico
# plt.plot(y_test, model_ytest,'.') # Dados experimental na forma de ponto
# plt.xlabel('ytest_experimentais') # Texto para o eixo x
# plt.ylabel('ytest_modelo')        # Texto para o eixo y
# plt.grid()                    # Mostra a grade
# plt.legend()                  # Legenda do gráfico
# plt.show()




pass
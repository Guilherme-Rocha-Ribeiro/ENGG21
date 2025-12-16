import kagglehub
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# Lendo os dados do kaggle
path = kagglehub.dataset_download("yasserh/housing-prices-dataset")
print("Path to dataset files:", path)
data = pd.read_csv(path + "/Housing.csv")
data.head() 



from sklearn.preprocessing import LabelEncoder
# Crie um LabelEncoder
le = LabelEncoder()
data_encoded = data.copy()
# Itere pelas colunas do DataFrame
for col in data_encoded.columns:
    # Verifique se a coluna não é numérica
    if not pd.api.types.is_numeric_dtype(data_encoded[col]):
        print(f"Coluna: {col}")
        # Aplique o LabelEncoder à coluna
        data_encoded[col] = le.fit_transform(data_encoded[col])

# Exibindo os dados codificados
data_encoded.head()


data = data.replace({'yes':1, 'no':0, 'semi-furnished':2, 'furnished':1, 'unfurnished':0})





features = data.drop(columns=['price'])
target = data['price']

X = features
y = target

# # Correlação
# plt.figure(figsize=(10, 8))
# sns.heatmap(data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
# plt.title('Matriz de Correlação entre Variáveis')
# plt.show()
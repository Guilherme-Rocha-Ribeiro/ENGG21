import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as tfk

plt.style.use(['seaborn-v0_8-paper'])

path = kagglehub.dataset_download("yasserh/breast-cancer-dataset")
print("Path to dataset files:", path)
df = pd.read_csv(path + '/breast-cancer.csv')

df=df.dropna()

print(df)



X = df[['radius_mean', 'texture_mean', 'perimeter_mean',

       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',

       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',

       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',

       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',

       'fractal_dimension_se', 'radius_worst', 'texture_worst',

       'perimeter_worst', 'area_worst', 'smoothness_worst',

       'compactness_worst', 'concavity_worst', 'concave points_worst',

       'symmetry_worst', 'fractal_dimension_worst']]



Y = df['diagnosis']

Y = Y.map({'M': 0, 'B': 1})

uni, counts = np.unique(Y,return_counts=True)

print(uni)

print(counts)

w0 = (1/counts[0])/(1/counts[0]+1/counts[1])

w1 = (1/counts[1])/(1/counts[0]+1/counts[1])

print(w0,w1)

#@title Rede neural

best_rna = []

loss_best = 100

for nn in range(1,20):

  rna = tfk.models.Sequential()

  # Camada de entrada

  n_col = X.shape[1] # numero de variveis de entradas

  rna.add(tfk.layers.Input(shape=(n_col,)))

  

  # Camada intermediárias

  rna.add(tfk.layers.Dense(nn,activation='relu'))

  #rna.add(tfk.layers.Dense(5,activation='relu'))



  # Camada de saída

  rna.add(tfk.layers.Dense(1,activation='sigmoid'))



  # Compilar

  opt = tfk.optimizers.Adam(learning_rate=0.01)

  rna.compile(optimizer=opt,

              loss='binary_crossentropy')

  rna.summary()



  # Treinar a rede

  treinamento = rna.fit(X,Y,epochs=500,

                        verbose=0,

                        class_weight={0:w0, 1:w1})

  if treinamento.history['loss'][-1] < loss_best:

    loss_best = treinamento.history['loss'][-1]

    best_rna = rna

plt.plot(treinamento.history['loss'])

plt.xlabel('Épocas')

plt.ylabel('Erro')

plt.show()





Y_pred = best_rna.predict(X)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

Y_pred[Y_pred>=0.5] = 1

Y_pred[Y_pred<0.5] = 0

cm = confusion_matrix(Y,Y_pred)

print(cm)

plt.figure()

disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot(cmap=plt.cm.Blues)

plt.xlabel('Classe predita')

plt.ylabel('Classe real')

plt.show()
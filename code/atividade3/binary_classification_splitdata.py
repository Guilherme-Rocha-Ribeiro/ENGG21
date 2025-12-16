import kagglehub
# https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as tf_keras
from sklearn.model_selection import train_test_split 

plt.style.use(['seaborn-v0_8-paper'])

path = kagglehub.dataset_download("yasserh/breast-cancer-dataset")
print("Path to dataset files:", path)

df = pd.read_csv(path + '/breast-cancer.csv')
df=df.dropna()
# print(df)



features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
            'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean','radius_se', 'texture_se', 'perimeter_se', 'area_se', 
            'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se','fractal_dimension_se', 'radius_worst', 
            'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst',
            'symmetry_worst', 'fractal_dimension_worst']

target = 'diagnosis'


X = df[features]
Y = df[target]


Y = Y.map({'M': 0, 'B': 1})

uni, counts = np.unique(Y,return_counts=True)
print(uni)
print(counts)
w0 = (1/counts[0])/(1/counts[0]+1/counts[1])
w1 = (1/counts[1])/(1/counts[0]+1/counts[1])
print(w0,w1)


X_train, X_validation, Y_train, Y_validation = train_test_split(X, # input 
                                                                Y, # target 
                                                                stratify=Y,    # Keeps the same class percentage in both train and validation sets.
                                                                test_size=0.3) # Sets the percentage of the test data
loss_best = 100

for nn in range(1, 20):
  print(f"--- Testing with {nn} neurons ---")
  rna = tf_keras.models.Sequential()
  n_col = X_train.shape[1] # Use X_train shape
  rna.add(tf_keras.layers.Input(shape=(n_col,)))
  
  rna.add(tf_keras.layers.Dense(nn, activation='relu'))
  
  # Camada de saída (Using the standard binary method)
  """
  The output layer is just the last layer you .add().
  Keras the sequence defines the flow. Data goes in the first layer, passes through any middle layers, and comes out the final one.
  """
  rna.add(tf_keras.layers.Dense(1, activation='sigmoid'))

  # Compilar
  opt = tf_keras.optimizers.Adam(learning_rate=0.01)
  rna.compile(optimizer=opt,
              loss='binary_crossentropy') 

  # Treinar a rede
  treinamento = rna.fit(X_train, Y_train, # Train on the 70%
                        epochs=500,
                        verbose=0,
                        class_weight={0: w0, 1: w1}, # Cannot use strings as dict key, 
                        validation_data=(X_validation, Y_validation)) # Test on the 30% 
  
  
  # 'val_loss' is the loss on the 30% validation set
  current_val_loss = treinamento.history['val_loss'][-1]
  
  if current_val_loss < loss_best:
    print(f"Neurons: {nn}, Val Loss: {current_val_loss}")
    loss_best = current_val_loss
    best_rna = rna



print("---")
print(f"Best model {best_rna.layers[0].units} neurons.")
    

plt.plot(treinamento.history['loss'])
plt.xlabel('Épocas')
plt.ylabel('Erro')
plt.show()


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# --- Predict ONLY on the validation set ---
Y_pred_proba = best_rna.predict(X_validation) # Predict on X_val, not X

# --- Convert probabilities to classes (0 or 1) ---
"""
Convert probabilities to binary classes in two steps:
1. (Y_pred_proba >= 0.5): Creates a boolean array (e.g., [False, True])
2. .astype(int): Converts booleans to integers (False=0, True=1)
"""
Y_pred_classes = (Y_pred_proba >= 0.5).astype(int) 

# --- Compare the predictions (Y_pred_classes) to the true labels (Y_val) ---
cm = confusion_matrix(Y_validation, Y_pred_classes) 

print("--- Confusion Matrix ---")
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.xlabel('Classe predita')
plt.ylabel('Classe real')
plt.show()

# A classification report gives you precision, recall, and f1-score
print("\n--- Classification Report ---")
print(classification_report(Y_validation, Y_pred_classes))
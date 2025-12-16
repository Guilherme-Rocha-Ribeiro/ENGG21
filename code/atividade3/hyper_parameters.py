import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import optuna



# Define Objective function







f1_metric = tf.keras.metrics.F1Score(average='micro')

def objective(trial):
    rna = tf.keras.Sequential()
    rna.add(tf.keras.layers.Input(shape=(28,28)))
    rna.add(tf.keras.layers.Flatten())
    # Hidden layers
    n_layers = trial.suggest_int("ln", 2, 6)
    neurons_size = trial.suggest_categorical("ls", [16, 32, 64, 128])
    # batch_sizes = trial.suggest_categorical("bs", [32, 64, 128])  # Atualiza o gradiente a cada bs valores
    for _ in range(n_layers):
        rna.add(tf.keras.layers.Dense(neurons_size, activation="relu"))
        rna.add(tf.keras.layers.Dropout(0.3))
    
    # Output layer
    rna.add(tf.keras.layers.Dense(10, activation='softmax'))   
    
    # Training
    rna.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy', f1_metric])
    
    training = rna.fit(
        train_images, train_labels,
        epochs=200,  
        verbose=0,
        validation_split=0.3,
        # batch_size=batch_sizes,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)]
    )
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(training.history['loss'], label='loss')
    plt.plot(training.history['val_loss'], label='val_loss')
    plt.legend()
    plt.xlabel('Épocas')
    plt.ylabel('Erro')

    plt.subplot(1, 2, 2)
    plt.plot(training.history['accuracy'], label='Training Accuracy')
    plt.plot(training.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.xlabel('Épocas')
    plt.ylabel('Accuracy')

    plt.show()
        
    return min(training.history['val_loss'][-1])

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
    
    















#@title Importando os dados
digits_mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = digits_mnist.load_data()


# Definindo os nomes das classes
class_names = ['0', '1', '2', '3', '4','5','6','7','8','9']
# Gráfico com a figura de 0 a 255
plt.figure()
plt.imshow(train_images[5],cmap='gray')
plt.colorbar()
plt.grid(False)
plt.show()
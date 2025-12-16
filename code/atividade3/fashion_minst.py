#@title Importando bibliotecas
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


dataset = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = dataset.load_data()

class_names = {
    0: 'T-shirt/top',
    1: 'Trouser', 
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}
class myCallback(tf.keras.callbacks.Callback):
 def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.95):
        print("\nReached 95% accuracy so cancelling training!")
        self.model.stop_training = True

callbacks = myCallback()
# Scale values to a range of 0 to 1 by dividing by the max value, in this case its 255
train_images = train_images / 255.0
test_images = test_images / 255.0

train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10) # 
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)   #






# Neural Network
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=(28,28)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))    

# Compile

f1_metric = tf.keras.metrics.F1Score(
    name='f1_score')


model.compile(optimizer='sgd', 
            loss='categorical_crossentropy',
            metrics=['accuracy', f1_metric]) # Mean Absolute Error | Root MSE

training = model.fit(train_images, train_labels,
          epochs=50,
          verbose=1,
          validation_split=0.3)

current_val_loss = training.history['val_loss'][-1]

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(training.history['loss'], label='loss')
plt.plot(training.history['val_loss'], label='val_loss')
plt.legend()
plt.xlabel('Épocas')
plt.ylabel('Erro')

plt.subplot(1, 3, 2)
plt.plot(training.history['f1_score'], label='f1-score')
plt.plot(training.history['val_f1_score'], label='Validation F1-Score')
plt.legend()
plt.xlabel('Épocas')
plt.ylabel('Accuracy')

plt.subplot(1, 3, 3)
plt.plot(training.history['accuracy'], label='accuracy')
plt.plot(training.history['val_accuracy'], label='Validation accuracy')
plt.legend()
plt.xlabel('Épocas')
plt.ylabel('Accuracy')

plt.show()
    

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

y = model.predict(test_images)
# Posição de maior probabilidade
preds = np.argmax(y, axis=1)

# Grafico
plt.figure()
cm = confusion_matrix(test_labels, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names.keys())
disp.plot()


print(classification_report(test_labels, preds, target_names=class_names.values()))



pass
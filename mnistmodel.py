import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#normalizin and converting to float 32 it's faster on gpu/cpu
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    x_train, y_train,
    epochs=40,
    batch_size=128,
    validation_data=(x_test, y_test),
    callbacks=[early_stop]
)

test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)

pred = model.predict(x_test)
print("Predicted:", np.argmax(pred[0]), "Actual:", y_test[0])

model.summary()

model.save("mnist_digit_model.keras")
print("model saved successfully!")

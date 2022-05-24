from tensorflow import keras

import tensorflow as tf

model = tf.keras.models.load_model('mnist.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("converted_mnist.tflite", "wb").write(tflite_model)

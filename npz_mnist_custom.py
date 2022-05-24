import glob 
import PIL 
import cv2 
from PIL import Image
import numpy as np 
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

train_data = 29211 #1300 # 데이터 수 
test_data = 3612
img_size = 300 #256 # 이미지의 사이즈 (정사각형으로 간주) 
color = 1 # 컬러는 3, 흑백은 1 
batch_size = 32 # 배치사이즈 


_x_train = np.load('x_train.npz')
_x_test = np.load('x_test.npz')

x_train = _x_train['x']
y_train = _x_train['y']

x_test = _x_test['x']
y_test = _x_test['y']

print(x_train.shape) # (60000, 28, 28) 
print(y_train.shape) # (60000, )
print(x_test.shape)
print(y_test.shape)


model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.summary()


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

checkpoint_path = "tmd_checkpoint.ckpt"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, 
                             save_weights_only=True, 
                             save_best_only=True, 
                             monitor='val_loss', 
                             verbose=1)

model.fit(x_train, y_train,
          epochs=5,
          validation_data=(x_test, y_test),
          callbacks=[checkpoint],
          )

model.save('mnist.h5')

model.load_weights(checkpoint_path)

model.evaluate(x_test, y_test)

import glob 
import PIL 
import cv2 
from PIL import Image
import numpy as np 

train = './x_train'
test  = './x_test'
#paths = [ './0', './1', './2', './3', './4', './5','./6', './7', './8', './9'] # './10', './11', './12', './13', './14', './15','./16', './17', './18', './19', './20' ]
paths = [ '/1', '/2', '/3', '/4', '/5','/6', '/7', '/8', '/9', '/10', '/11', '/12', '/13', '/14', '/15','/16', '/17', '/18', '/19', '/>', '/<']

train_data = 10606 #29211 #1300 # 데이터 수 
test_data = 4558 #3612
img_size = 150 #300 #256 # 이미지의 사이즈 (정사각형으로 간주) 
color = 1 # 컬러는 3, 흑백은 1 
batch_size = 32 # 배치사이즈 

try :
    # 곱셈의 형태로 만들기 

    x_train = np.zeros(train_data*img_size*img_size*color, dtype=np.int32).reshape(train_data, img_size, img_size ) #, color) 
    y_train = np.zeros(train_data, dtype=np.int32) 

    x_test = np.zeros(test_data*img_size*img_size*color, dtype=np.int32).reshape(test_data, img_size, img_size )
    y_test = np.zeros(test_data, dtype=np.int32) 
except Exception as e:
    print(e)
    

# # 덧셈의 형태로 만들기 
# x = np.zeros((number_of_data,) + (img_size, img_size) + (color,), dtype="float32") 
# y = np.zeros((number_of_data,) + (1,), dtype="uint8")


file_path = '' 
i=0
for v in range(0, len(paths)) : #paths: #path = "./img"
   
    file_path = train+paths[v] + '/' #path+'/'
    print(file_path, paths[v])
    # PIL을 활용한 방법 
    #i = 0 
    for file in sorted(glob.glob(file_path + '*.png')) : 
        #print(file)
        img = np.array(Image.open(file), dtype=np.int32) # PIL로 불러들인 이미지를 numpy array에 입력 
        x_train[i, :, :] = img #, :] = img # i번째에 이미지 픽셀값 입력 
        y_train[i] = v # i번째에 normal을 0으로 라벨링 
        #print(paths[v], v)
        i += 1 

#x_train = x_train/255.0

file_path = '' 
i=0
for v in range(0, len(paths)) : #paths: #path = "./img"
   
    file_path = test+paths[v] + '/' #path+'/'
    print(file_path, paths[v])
    # PIL을 활용한 방법 
    #i = 0 
    for file in sorted(glob.glob(file_path + '*.png')) : 
        print(file)
        img = np.array(Image.open(file), dtype=np.int32) # PIL로 불러들인 이미지를 numpy array에 입력 
        x_test[i, :, :] = img #, :] = img # i번째에 이미지 픽셀값 입력 
        y_test[i] = v # i번째에 normal을 0으로 라벨링 
        #print(paths[v], v)
        i += 1 

#x_test = x_test /255.0

print(x_train.shape) # (60000, 28, 28) 
print(y_train.shape) # (60000, )
print(x_test.shape)
print(y_test.shape)

x_train = x_train.reshape((train_data, img_size, img_size, color))
x_test = x_test.reshape((test_data, img_size, img_size, color))
x_train, x_test = x_train / 255.0, x_test / 255.0

print(x_train.shape) # (60000, 28, 28) 
print(y_train.shape) # (60000, )
print(x_test.shape)
print(y_test.shape)


import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

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

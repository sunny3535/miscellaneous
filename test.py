# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
#from tensorflow.contrib import lite
# Helper libraries
import numpy as np
from PIL import Image
from keras.preprocessing.image import img_to_array
import time

interpreter = tf.lite.Interpreter(model_path='converted_mnist.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# check the type of the input tensor
floating_model = input_details[0]['dtype'] == np.float32

# NxHxWxC, H:1, W:2
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
print(height, width)

img2 = Image.open('2.jpg').convert('L')
img = img2.resize((width, height))
input_data = img_to_array(img)
input_data = np.expand_dims(input_data, axis=0)

#input_data[:,:] = np.array(img2, dtype=np.int8) # PIL로 불러들인 이미지를 numpy array에 입력 
#input_data = input_data.reshape((width, height, 1))
# add N dim
#input_data = np.expand_dims(img, axis=0)
print(input_data)
print(input_data.shape)
#input_data = input_data.reshape((width, height, 1))
#if floating_model:
     #input_data = (np.float32(input_data) - img.input_mean) / img.input_std
#input_data = np.float32(input_data)


# img = np.array(Image.open(file), dtype=np.int8) # PIL로 불러들인 이미지를 numpy array에 입력 
#         x_train[i, :, :] = img #, :] = img # i번째에 이미지 픽셀값 입력 

interpreter.set_tensor(input_details[0]['index'], input_data)

start_time = time.time()
interpreter.invoke()
stop_time = time.time()

output_data = interpreter.get_tensor(output_details[0]['index'])
results = np.squeeze(output_data)

labels = [ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
top_k = results.argsort()[-5:][::-1]
#labels = load_labels(args.label_file)
print(top_k)
for i in top_k:
    #if floating_model:
    #  print('{:08.6f}: {}'.format(float(results[i]), labels[i]))
    #else:
    print(i)
    print('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]))

print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))

print(labels[int(results[0])])
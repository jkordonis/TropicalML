
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from keras.layers import Dense, Conv2D, Activation, MaxPool2D, Flatten, Dropout, BatchNormalization
from keras.optimizers import RMSprop,Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
import math
from keras.optimizers import RMSprop


Img_shape = 28
Num_classes = 10
test_size = 0.25
random_state = 1234
No_epochs = 100
Batch_size = 128

Dire="C:/TensorFlowExamples/runn2/TensorFlowExamples_transf/fashion_mnist/"

train_dataset = pd.read_csv(Dire+"fashion-mnist_train.csv")
test_dataset = pd.read_csv(Dire+"fashion-mnist_test.csv")

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal',      'Shirt',   'Sneaker',  'Bag',   'Ankle boot']

def data_preprocessing(raw):
    label = tf.keras.utils.to_categorical(raw.label, 10)
    num_images = raw.shape[0]
    x_as_array = raw.values[:,1:]
    x_shaped_array = x_as_array.reshape(num_images, 28, 28, 1)
    image = x_shaped_array / 255
    return image, label

X, y = data_preprocessing(train_dataset)
X_test, y_test = data_preprocessing(test_dataset)


X_test=np.load('C:/TensorFlowExamples/runn2/TensorFlowExamples_transf/fashion_mnist/Interm_Layer_test.npy')
X_train=np.load('C:/TensorFlowExamples/runn2/TensorFlowExamples_transf/fashion_mnist/Interm_Layer_train.npy')



model = Sequential()
model.add(Dropout(0.4))
model.add(BatchNormalization())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.4))
model.add(BatchNormalization())
model.add(Dense(500, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))


model.compile(optimizer='adam',
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])



train_model = model.fit(X_train, y,
                  batch_size=Batch_size,
                  epochs=20,
                  verbose=1)

score = model.evaluate(X_test, y_test, steps=math.ceil(10000/32))

# checking the test loss and test accuracy
print('Test loss:', score[0])
print('Test accuracy:', score[1])

import gzip, os.path
import json
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Reshape
import numpy as np




#print(data)
#print(len(data),len(data[0]),len(data[0][0]),len(data[0][0][0]))
#print(data[0][0])
#print()
#print(data[0][1])


def calculate_error(Y_pred,Y_test):
  res = [0]*6
  for i in range(len(Y_pred)):
    val = abs(Y_pred[i]-Y_test[i])
    if val <= 0.05:
      res[0]+=1
    elif val <= 0.10:
      res[1]+=1
    elif val <= 0.20:
      res[2]+=1
    elif val <= 0.35:
      res[3]+=1
    elif val <= 0.50:
      res[4]+=1
    else:
      res[5]+=1
  return [x / len(Y_pred) for x in res]





model = Sequential([
    Conv2D(128, (5, 5), padding='same', activation = 'relu', data_format='channels_last', input_shape=(9,9,11)),
    Dropout(rate=0.5),
    BatchNormalization(),
    Conv2D(128, (3, 3), padding='same', activation = 'relu', data_format='channels_last'),
    Dropout(rate=0.5),
    BatchNormalization(),
    Conv2D(128, (3, 3), padding='same', activation = 'relu', data_format='channels_last'),
    Dropout(rate=0.5),
    BatchNormalization(),
    Conv2D(128, (3, 3), padding='same', activation = 'relu', data_format='channels_last'),
    Dropout(rate=0.5),
    BatchNormalization(),
    Flatten(),
    Dense(2048, activation = 'relu'),
    Dropout(rate=0.5),
    Dense(1024, activation = 'relu'),
    Dense(512, activation = 'relu'),
    Dense(256, activation = 'relu'),
    Dense(81, activation = 'softmax')
])

model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

model.summary()

model.save('.')

#res = calculate_error(model.predict(X_test),Y_test)
#for i in range(6):
#  print("%.2f" % (res[i]))


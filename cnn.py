from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Lambda
from keras.layers import Embedding
from keras.layers import Convolution1D,MaxPooling1D, Flatten
from keras.datasets import imdb
from keras import backend as K
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import Normalizer,LabelEncoder
from keras.models import Sequential
from keras.layers import Convolution1D, Dense, Dropout, Flatten, MaxPooling1D
from keras.utils import np_utils
import numpy as np
import h5py
from keras import callbacks
from keras.layers import LSTM, GRU, SimpleRNN
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn import metrics


traindata = pd.read_csv('kddtrain.csv', header=None)
testdata = pd.read_csv('kddtest.csv', header=None)


X = traindata.iloc[1:,0:41].values
label_encoder_x = LabelEncoder()
X[:,1] = label_encoder_x.fit_transform(X[:,1])
X[:,2] = label_encoder_x.fit_transform(X[:,2])
X[:,3] = label_encoder_x.fit_transform(X[:,3])
#print(X)
print("----------------")
Y = traindata.iloc[1:,41].values
#print(Y)
Y = label_encoder_x.fit_transform(Y)
#print(Y)

C = testdata.iloc[1:,41]
#print(C)
C = label_encoder_x.fit_transform(C)
#print(C)
T = testdata.iloc[1:,0:41].values
#print(T)
print("-----------------")
T[:,1] = label_encoder_x.fit_transform(T[:,1])
T[:,2] = label_encoder_x.fit_transform(T[:,2])
T[:,3] = label_encoder_x.fit_transform(T[:,3])
#print(T)

scaler = Normalizer().fit(X)
trainX = scaler.transform(X)

scaler = Normalizer().fit(T)
testT = scaler.transform(T)

y_train = np.array(Y)
y_test = np.array(C)


# reshape input to be [samples, time steps, features]
X_train = np.reshape(trainX, (trainX.shape[0],trainX.shape[1],1))
X_test = np.reshape(testT, (testT.shape[0],testT.shape[1],1))


lstm_output_size = 128

cnn = Sequential()
cnn.add(Convolution1D(64, 3, border_mode="same",activation="relu",input_shape=(41, 1)))
cnn.add(Convolution1D(64, 3, border_mode="same", activation="relu"))
cnn.add(MaxPooling1D(pool_length=(2)))
cnn.add(Convolution1D(128, 3, border_mode="same", activation="relu"))
cnn.add(Convolution1D(128, 3, border_mode="same", activation="relu"))
cnn.add(MaxPooling1D(pool_length=(2)))
cnn.add(Flatten())
cnn.add(Dense(128, activation="relu"))
cnn.add(Dropout(0.5))
cnn.add(Dense(1, activation="sigmoid"))

print("CNN Layers for Detecting anomaly or normal")
cnn.summary()

# define optimizer and objective, compile cnn

cnn.compile(loss="binary_crossentropy", optimizer="adam",metrics=['accuracy'])

# train
cnn.fit(X_train, y_train, nb_epoch=5,validation_data=(X_test, y_test)) #change nb_epoch to your 
cnn.save("cnn_model.hdf5") # saving model here
y_pred = cnn.predict_classes(X_test)
np.savetxt('predicted3.txt', y_pred, fmt='%01d')
cnn.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
loss, accuracy = cnn.evaluate(X_test, y_test)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))

# loss, accuracy = cnn.evaluate(X_test, y_test)
# print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))

# y_pred = cnn.predict_classes(X_test)
# np.savetxt('res/expected3.txt', y_test, fmt='%01d')
# np.savetxt('res/predicted3.txt', y_pred, fmt='%01d')

# accuracy = accuracy_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred , average="binary")
# precision = precision_score(y_test, y_pred , average="binary")
# f1 = f1_score(y_test, y_pred, average="binary")

# print("confusion matrix")
# print("----------------------------------------------")
# print("accuracy")
# print("%.6f" %accuracy)
# print("racall")
# print("%.6f" %recall)
# print("precision")
# print("%.6f" %precision)
# print("f1score")
# print("%.6f" %f1)
# cm = metrics.confusion_matrix(y_test, y_pred)
# print("==============================================")
'''
Source codes for Hands-On Deep Learning Architectures with Python (Packt Publishing)
Chapter 6 Recurrent Neural Networks
Author: Yuxi (Hayden) Liu
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


raw_data = pd.read_csv('^DJI.csv')
raw_data.head()
data = raw_data.Close.values
len(data)

plt.plot(data)
plt.xlabel('Time period')
plt.ylabel('Price')
plt.show()


def generate_seq(data, window_size):
    """
    Transform input series into input sequences and outputs based on a specified window size
    @param data: input series
    @param window_size: int
    @return: numpy array of input sequences, numpy array of outputs
    """
    X, Y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        Y.append(data[i])
    return np.array(X),np.array(Y)


window_size = 10
X, Y = generate_seq(data, window_size)
X.shape
Y.shape


train_ratio = 0.7
val_ratio = 0.1
train_n = int(len(Y) * train_ratio)

X_train = X[:train_n]
Y_train = Y[:train_n]

X_test = X[train_n:]
Y_test = Y[train_n:]


def scale(X, Y):
    """
    Scaling the prices within each window
    @param X: input series
    @param Y: outputs
    @return: scaled input series and outputs
    """
    X_processed, Y_processed = np.copy(X), np.copy(Y)
    for i in range(len(X)):
        x = X[i, -1]
        X_processed[i] /= x
        Y_processed[i] /= x
    return X_processed, Y_processed


def reverse_scale(X, Y_scaled):
    """
    Convert the scaled outputs to the original scale
    @param X: original input series
    @param Y_scaled: scaled outputs
    @return: outputs in original scale
    """
    Y_original = np.copy(Y_scaled)
    for i in range(len(X)):
        x = X[i, -1]
        Y_original[i] *= x
    return Y_original


X_train_scaled, Y_train_scaled = scale(X_train, Y_train)
X_test_scaled, Y_test_scaled = scale(X_test, Y_test)



from keras.models import Sequential
from keras.layers import Dense, GRU
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras import optimizers

model = Sequential()
model.add(GRU(256, input_shape=(window_size, 1)))
model.add(Dense(1))

optimizer = optimizers.RMSprop(lr=0.0006, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(loss='mean_squared_error', optimizer=optimizer)


tensorboard = TensorBoard(log_dir='./logs/run1/', write_graph=True, write_images=False)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='min')
model_file = "weights/best_model.hdf5"
checkpoint = ModelCheckpoint(model_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

model.fit(X_train_reshaped, Y_train_scaled, validation_data=(X_test_reshaped, Y_test_scaled),
          epochs=1, batch_size=100, verbose=1, callbacks=[tensorboard, early_stop, checkpoint])


from keras.models import load_model
model = load_model(model_file)

pred_train_scaled = model.predict(X_train_reshaped)
pred_test_scaled = model.predict(X_test_reshaped)

pred_train = reverse_scale(X_train, pred_train_scaled)
pred_test = reverse_scale(X_test, pred_test_scaled)


plt.plot(Y)
plt.plot(np.concatenate([pred_train, pred_test]))
plt.xlabel('Time period')
plt.ylabel('Price')
plt.legend(['original series','prediction'],loc='center left')
plt.show()



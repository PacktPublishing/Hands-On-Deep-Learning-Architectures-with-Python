'''
Source codes for Hands-On Deep Learning Architectures with Python (Packt Publishing)
Chapter 3 Restricted Boltzmann Machines and Autoencoders
Author: Yuxi (Hayden) Liu
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import optimizers
import keras.backend as K


data = pd.read_csv("creditcard.csv").drop(['Time'], axis=1)
print(data.shape)

print('Number of fraud samples: ', sum(data.Class == 1))
print('Number of normal samples: ', sum(data.Class == 0))


scaler = StandardScaler()
data['Amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))



np.random.seed(1)
data_train, data_test = train_test_split(data, test_size=0.2)


data_test = data_test.append(data_train[data_train.Class == 1], ignore_index=True)
data_train = data_train[data_train.Class == 0]

X_train = data_train.drop(['Class'], axis=1).values

X_test = data_test.drop(['Class'], axis=1).values
Y_test = data_test['Class']

input_size = 29
hidden_size = 40

input_layer = Input(shape=(input_size,))
encoder = Dense(hidden_size, activation="relu")(input_layer)
decoder = Dense(input_size)(encoder)
contractive_ae = Model(inputs=input_layer, outputs=decoder)
print(contractive_ae.summary())

optimizer = optimizers.Adam(lr=0.0003)


factor = 1e-5
def contractive_loss(y_pred, y_true):
    mse = K.mean(K.square(y_true - y_pred), axis=1)
    W = K.variable(value=contractive_ae.layers[1].get_weights()[0])
    W_T = K.transpose(W)
    W_T_sq_sum = K.sum(W_T ** 2, axis=1)
    h = contractive_ae.layers[1].output
    contractive = factor * K.sum((h * (1 - h)) ** 2 * W_T_sq_sum, axis=1)
    return mse + contractive


contractive_ae.compile(optimizer=optimizer, loss=contractive_loss)

tensorboard = TensorBoard(log_dir='./logs/run4/', write_graph=True, write_images=False)

model_file = "model_contractive_ae.h5"
checkpoint = ModelCheckpoint(model_file, monitor='loss', verbose=1, save_best_only=True, mode='min')

num_epoch = 30
batch_size = 64
contractive_ae.fit(X_train, X_train, epochs=num_epoch, batch_size=batch_size, shuffle=True, validation_data=(X_test, X_test),
                   verbose=1, callbacks=[checkpoint, tensorboard])

recon = contractive_ae.predict(X_test)

recon_error = np.mean(np.power(X_test - recon, 2), axis=1)




from sklearn.metrics import (precision_recall_curve, auc)


precision, recall, th = precision_recall_curve(Y_test, recon_error)

area = auc(recall, precision)
print('Area under precision-recall curve:', area)


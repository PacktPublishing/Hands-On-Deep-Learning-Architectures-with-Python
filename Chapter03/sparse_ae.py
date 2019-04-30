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


data = pd.read_csv("creditcard.csv").drop(['Time'], axis=1)
print(data.shape)

print('Number of fraud samples: ', sum(data.Class == 1))
print('Number of normal samples: ', sum(data.Class == 0))


scaler = StandardScaler()
data['Amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))



np.random.seed(1)
data_train, data_test = train_test_split(data, test_size=0.95)


data_test = data_test.append(data_train[data_train.Class == 1], ignore_index=True)
data_train = data_train[data_train.Class == 0]

X_train = data_train.drop(['Class'], axis=1).values

X_test = data_test.drop(['Class'], axis=1).values
Y_test = data_test['Class']

input_size = 29
hidden_size = 40


from keras import regularizers

hidden_sizes = [80, 40, 80]

input_layer = Input(shape=(input_size,))
encoder = Dense(hidden_sizes[0], activation="relu", activity_regularizer=regularizers.l1(3e-5))(input_layer)
encoder = Dense(hidden_sizes[1], activation="relu")(encoder)
decoder = Dense(hidden_sizes[2], activation='relu')(encoder)
decoder = Dense(input_size)(decoder)
sparse_ae = Model(inputs=input_layer, outputs=decoder)
print(sparse_ae.summary())



optimizer = optimizers.Adam(lr=0.0008)
sparse_ae.compile(optimizer=optimizer, loss='mean_squared_error')

tensorboard = TensorBoard(log_dir='./logs/run3/', write_graph=True, write_images=False)

model_file = "model_sparse_ae.h5"
checkpoint = ModelCheckpoint(model_file, monitor='loss', verbose=1, save_best_only=True, mode='min')

num_epoch = 30
batch_size = 64
sparse_ae.fit(X_train, X_train, epochs=num_epoch, batch_size=batch_size, shuffle=True, validation_data=(X_test, X_test),
              verbose=1, callbacks=[checkpoint, tensorboard])

recon = sparse_ae.predict(X_test)

recon_error = np.mean(np.power(X_test - recon, 2), axis=1)


from sklearn.metrics import (precision_recall_curve, auc)

precision, recall, th = precision_recall_curve(Y_test, recon_error)
area = auc(recall, precision)
print('Area under precision-recall curve:', area)



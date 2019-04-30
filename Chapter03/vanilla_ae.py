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
import matplotlib.pyplot as plt


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
ae = Model(inputs=input_layer, outputs=decoder)
print(ae.summary())

optimizer = optimizers.Adam(lr=0.0001)
ae.compile(optimizer=optimizer, loss='mean_squared_error')

tensorboard = TensorBoard(log_dir='./logs/run1/', write_graph=True, write_images=False)

model_file = "model_ae.h5"
checkpoint = ModelCheckpoint(model_file, monitor='loss', verbose=1, save_best_only=True, mode='min')

num_epoch = 30
batch_size = 64
ae.fit(X_train, X_train, epochs=num_epoch, batch_size=batch_size, shuffle=True, validation_data=(X_test, X_test),
       verbose=1, callbacks=[checkpoint, tensorboard])

recon = ae.predict(X_test)

recon_error = np.mean(np.power(X_test - recon, 2), axis=1)




from sklearn.metrics import (roc_auc_score, precision_recall_curve, auc, confusion_matrix)

roc_auc = roc_auc_score(Y_test, recon_error)
print('Area under ROC curve:', roc_auc)

precision, recall, th = precision_recall_curve(Y_test, recon_error)

plt.plot(recall, precision, 'b')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()

area = auc(recall, precision)
print('Area under precision-recall curve:', area)


plt.plot(th, precision[1:], 'k')
plt.plot(th, recall[1:], 'b', label='Threshold-Recall curve')
plt.title('Precision (black) and recall (blue) for different threshold values')
plt.xlabel('Threshold of reconstruction error')
plt.ylabel('Precision or recall')
plt.show()


threshold = .000001
Y_pred = [1 if e > threshold else 0 for e in recon_error]
conf_matrix = confusion_matrix(Y_test, Y_pred)
print(conf_matrix)

'''
Source codes for Hands-On Deep Learning Architectures with Python (Packt Publishing)
Chapter 6 Recurrent Neural Networks
Author: Yuxi (Hayden) Liu
'''

from keras.datasets import imdb

word_to_id = imdb.get_word_index()


max_words = 5000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words, skip_top=10, seed=42)

print(len(y_train), 'training samples')
print(len(y_test), 'testing samples')


from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Bidirectional
from keras import optimizers

maxlen = 500

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)



model = Sequential()
model.add(Embedding(max_words, 128, input_length=maxlen))
model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)))

model.add(Dense(1, activation='sigmoid'))

optimizer = optimizers.RMSprop(0.001)
model.compile(optimizer=optimizer,loss='binary_crossentropy', metrics=['accuracy'])

from keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='min')

hist = model.fit(x_train, y_train,
          batch_size=32,
          epochs=100,
          validation_data=[x_test, y_test], callbacks=[early_stop])

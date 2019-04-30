# importing the Sequential method in Keras
import keras
from keras.models import Sequential

# Importing the Dense layer which creates a layer of Deep Feedforward Network
from keras.layers import Dense, Activation, Flatten, Dropout

# getting the data as we did earlier
fashionObj = keras.datasets.fashion_mnist

(trainX, trainY), (testX, testY) = fashionObj.load_data()
print('train data x shape: ', trainX.shape)
print('test data x shape:', testX.shape)

print('train data y shape: ', trainY.shape)
print('test data y shape: ', testY.shape)


# Now we can directly jump to building model, we build in Sequential manner as discussed in Chapter 1
model = Sequential()

# the first layer we will use is to flatten the 2-d image input from (28,28) to 784
model.add(Flatten(input_shape = (28, 28)))

# adding first hidden layer with 512 units
model.add(Dense(512))

#adding activation to the output
model.add(Activation('relu'))

#using Dropout for Regularization
model.add(Dropout(0.2))

# adding our final output layer
model.add(Dense(10))

#softmax activation at the end
model.add(Activation('softmax'))

# normalising input data before feeding
trainX = trainX / 255
testX = testX / 255

# compiling model with optimizer and loss
model.compile(optimizer= 'Adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# training the model
model.fit(trainX, trainY, epochs = 5, batch_size = 64)

# evaluating the model on test data
evalu = model.evaluate(testX, testY)
print('Test Set average Accuracy: ', evalu[1])


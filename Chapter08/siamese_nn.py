'''
Source codes for Hands-On Deep Learning Architectures with Python (Packt Publishing)
Chapter 8 New Trends of Deep Learning
Author: Yuxi (Hayden) Liu
'''

import numpy as np
from PIL import Image

from keras import backend as K
from keras.layers import Activation
from keras.layers import Input, Lambda, Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
from keras.models import Sequential, Model




img = Image.open('./orl_faces/s1/1.pgm')
print(img.size)
# img.show()

image_size = [92, 112, 1]

def load_images_ids(path='./orl_faces'):
    id_image = {}
    for id in range(1, 41):
        id_image[id] = []
        for image_id in range(1, 11):
            img = Image.open('{}/s{}/{}.pgm'.format(path, id, image_id))
            img = np.array(img).reshape(image_size)
            id_image[id].append(img)
    return id_image


id_image = load_images_ids()




def siamese_network():
    seq = Sequential()
    nb_filter = 16
    kernel_size = 6
    # Convolution layer
    seq.add(Convolution2D(nb_filter, (kernel_size, kernel_size), input_shape=image_size, border_mode='valid'))
    seq.add(Activation('relu'))
    seq.add(MaxPooling2D(pool_size=(2, 2)))
    seq.add(Dropout(.25))
    # flatten
    seq.add(Flatten())
    seq.add(Dense(50, activation='relu'))
    seq.add(Dropout(0.1))
    return seq


img_1 = Input(shape=image_size)
img_2 = Input(shape=image_size)

base_network = siamese_network()
feature_1 = base_network(img_1)
feature_2 = base_network(img_2)


distance_function = lambda x: K.abs(x[0] - x[1])
distance = Lambda(distance_function, output_shape=lambda x: x[0])([feature_1, feature_2])
prediction = Dense(1, activation='sigmoid')(distance)

model = Model(input=[img_1, img_2], output=prediction)




from keras.losses import binary_crossentropy
from keras.optimizers import Adam
optimizer = Adam(lr=0.001)

model.compile(loss=binary_crossentropy, optimizer=optimizer)


np.random.seed(42)

def gen_train_data(n, id_image):
    X_1, X_2 = [], []
    Y = [1] * (n // 2) + [0] * (n // 2)
    # generate positive samples
    ids = np.random.choice(range(1, 41), n // 2)
    for id in ids:
        two_image_ids = np.random.choice(range(10), 2, False)
        X_1.append(id_image[id][two_image_ids[0]])
        X_2.append(id_image[id][two_image_ids[1]])
    # generate negative samples, by randomly selecting two images from two ids
    for _ in range(n // 2):
        two_ids = np.random.choice(range(1, 41), 2, False)
        two_image_ids = np.random.randint(0, 10, 2)
        X_1.append(id_image[two_ids[0]][two_image_ids[0]])
        X_2.append(id_image[two_ids[1]][two_image_ids[1]])
    X_1 = np.array(X_1).reshape([n] + image_size) / 255
    X_2 = np.array(X_2).reshape([n] + image_size) / 255
    Y = np.array(Y)
    return [X_1, X_2], Y



def gen_test_case(n_way):
    ids = np.random.choice(range(1, 41), n_way)
    id_1 = ids[0]
    image_1 = np.random.randint(0, 10, 1)[0]
    image_2 = np.random.randint(image_1 + 1, 9 + image_1, 1)[0] % 10
    X_1 = [id_image[id_1][image_1]]
    X_2 = [id_image[id_1][image_2]]
    for id_2 in ids[1:]:
        image_2 = np.random.randint(0, 10, 1)[0]
        X_1.append(id_image[id_1][image_1])
        X_2.append(id_image[id_2][image_2])
    X_1 = np.array(X_1).reshape([n_way] + image_size) / 255
    X_2 = np.array(X_2).reshape([n_way] + image_size) / 255
    return [X_1, X_2]




X_train, Y_train = gen_train_data(8000, id_image)

epochs = 10
model.fit(X_train, Y_train, validation_split=0.1, batch_size=64, verbose=1, epochs=epochs)


def knn(X):
    distances = [np.linalg.norm(x_1 - x_2) for x_1, x_2 in zip(X[0], X[1])]
    pred = np.argmin(distances)
    return pred


n_experiment = 1000

for n_way in [4, 9, 16, 25, 36, 40]:
    n_correct_snn = 0
    n_correct_knn = 0
    for _ in range(n_experiment):
        X_test = gen_test_case(n_way)
        pred = model.predict(X_test)
        pred_id = np.argmax(pred)
        if pred_id == 0:
            n_correct_snn += 1
        if knn(X_test) == 0:
            n_correct_knn += 1
    print('{}-way few shot learning accuracy: {}'.format(n_way, n_correct_snn / n_experiment))
    print('Baseline accuracy with knn: {}\n'.format(n_correct_knn / n_experiment))

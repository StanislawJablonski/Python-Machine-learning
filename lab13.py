from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
import cv2
import numpy as np

def load_image(filename):
    img = 255 - cv2.imread(filename, 0)
    img = cv2.resize(img, (28, 28))
    img = np.array(img)
    img = img.reshape(1, 28, 28, 1)
    img = img.astype('float32')
    return img / 255.0


def train():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape((60000, 28, 28, 1))
    x_test = x_test.reshape((10000, 28, 28, 1))
    x_train = x_train.astype('float') / 255
    x_test = x_test.astype('float') / 255
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    cnn = Sequential()
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Flatten())
    cnn.add(Dense(units=128, activation='relu'))
    cnn.add(Dense(units=10, activation='softmax'))
    cnn.compile(optimizer='adam', loss='categorical_crossentropy',
                metrics=['accuracy'])
    cnn.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)
    return cnn



model = train()

images = ['one', 'two', 'three']
for image in images:
    img = load_image(f'{image}.png')
    digit = model.predict(img)
    print(f'{image}: {np.argmax(digit[0])}')
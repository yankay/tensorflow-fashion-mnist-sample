#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import gzip
import numpy as np
import tensorflow as tf
from tensorflow import keras
print('TensorFlow version: {}'.format(tf.__version__))
dataset_path = "/root/data"
model_path = "./model/"
model_version =  "v1"

def load_data():
    files = [
        'train-labels-idx1-ubyte.gz',
        'train-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz'
    ]
    paths = []
    for fname in files:
        paths.append(os.path.join(dataset_path, fname))
    with gzip.open(paths[0], 'rb') as labelpath:
        y_train = np.frombuffer(labelpath.read(), np.uint8, offset=8)
    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)
    with gzip.open(paths[2], 'rb') as labelpath:
        y_test = np.frombuffer(labelpath.read(), np.uint8, offset=8)
    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)
    return (x_train, y_train),(x_test, y_test)

def train():
    (train_images, train_labels), (test_images, test_labels) = load_data()

    # scale the values to 0.0 to 1.0
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # reshape for feeding into the model
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    print('\ntrain_images.shape: {}, of {}'.format(train_images.shape, train_images.dtype))
    print('test_images.shape: {}, of {}'.format(test_images.shape, test_images.dtype))

    model = keras.Sequential([
    keras.layers.Conv2D(input_shape=(28,28,1), filters=8, kernel_size=3,
                        strides=2, activation='relu', name='Conv1'),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation=tf.nn.softmax, name='Softmax')
    ])
    model.summary()
    testing = False
    epochs = 5
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    logdir = "/training_logs"
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    model.fit(train_images,
        train_labels,
        epochs=epochs,
        callbacks=[tensorboard_callback],
    )
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('\nTest accuracy: {}'.format(test_acc))
    export_path = os.path.join(model_path, model_version)
    os.makedirs(model_path)
    print('export_path = {}\n'.format(export_path))
    tf.keras.models.save_model(
        model,
        export_path,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None
    )
    print('\nSaved model success')
if __name__ == '__main__':
    train()

import os
import cv2
import joblib
import numpy as np

import scikitplot
import seaborn as sns
from matplotlib import pyplot

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model


class CNNModel():

    def __init__(self):
        self.model = None
        self.trained = False

    def model_builder(self, in_shape, out_shape):
        model_in = Input(shape=in_shape, name="input_CNN")
        
        conv2d_1 = Conv2D(
            filters=64,
            kernel_size=(3,3),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_1'
        )(model_in)
        batchnorm_1 = BatchNormalization(name='batchnorm_1')(conv2d_1)
        conv2d_2 = Conv2D(
            filters=64,
            kernel_size=(3,3),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_2'
        )(batchnorm_1)
        batchnorm_2 = BatchNormalization(name='batchnorm_2')(conv2d_2)
        
        maxpool2d_1 = MaxPooling2D(pool_size=(2,2), name='maxpool2d_1')(batchnorm_2)
        dropout_1 = Dropout(0.35, name='dropout_1')(maxpool2d_1)

        conv2d_3 = Conv2D(
            filters=128,
            kernel_size=(3,3),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_3'
        )(dropout_1)
        batchnorm_3 = BatchNormalization(name='batchnorm_3')(conv2d_3)
        conv2d_4 = Conv2D(
            filters=128,
            kernel_size=(3,3),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_4'
        )(batchnorm_3)
        batchnorm_4 = BatchNormalization(name='batchnorm_4')(conv2d_4)
        
        maxpool2d_2 = MaxPooling2D(pool_size=(2,2), name='maxpool2d_2')(batchnorm_4)
        dropout_2 = Dropout(0.4, name='dropout_2')(maxpool2d_2)

        conv2d_5 = Conv2D(
            filters=256,
            kernel_size=(3,3),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_5'
        )(dropout_2)
        batchnorm_5 = BatchNormalization(name='batchnorm_5')(conv2d_5)
        conv2d_6 = Conv2D(
            filters=256,
            kernel_size=(3,3),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_6'
        )(batchnorm_5)
        batchnorm_6 = BatchNormalization(name='batchnorm_6')(conv2d_6)
        
        maxpool2d_3 = MaxPooling2D(pool_size=(2,2), name='maxpool2d_3')(batchnorm_6)
        dropout_3 = Dropout(0.5, name='dropout_3')(maxpool2d_3)

        flatten = Flatten(name='flatten')(dropout_3)
            
        dense_1 = Dense(
            128,
            activation='elu',
            kernel_initializer='he_normal',
            name='dense1'
        )(flatten)
        batchnorm_7 = BatchNormalization(name='batchnorm_7')(dense_1)

        model_out = Dropout(0.6, name='dropout_4')(batchnorm_7)

        model_out = Dense(out_shape, activation="softmax", name="out_layer")(model_out)
        self.model = Model(inputs=model_in, outputs=model_out, name="CNN")

    def train(self, X_train, y_train, validation_data, batch_size=24, epochs=50, callbacks=None, train_datagen=None):
        self.model_builder(X_train.shape[1:], y_train.shape[1])

        self.model.compile(
            loss="categorical_crossentropy",
            optimizer=optimizers.Adam(0.01),
            metrics=['accuracy']
        )

        if train_datagen is None:
            self.history = self.model.fit(
                X_train, y_train,
                validation_data = validation_data,
                batch_size = batch_size,
                epochs = epochs,
                callbacks = callbacks,
            )
        else:
            steps_per_epoch = len(X_train) / batch_size
            self.history = self.model.fit(
                train_datagen.flow(X_train, y_train, batch_size=batch_size),
                validation_data = validation_data,
                steps_per_epoch = steps_per_epoch,
                epochs = epochs,
                callbacks = callbacks,
            )

        self.trained = True

    def evaluate(self, X_test, y_test, save_evaluation_to=None):
        if self.trained:
            yhat_test = np.argmax(self.model.predict(X_test), axis=1)
            ytest_ = np.argmax(y_test, axis=1)
            test_accu = np.sum(ytest_ == yhat_test) / len(ytest_) * 100
            print(f"test accuracy: {round(test_accu, 4)} %\n\n")
            print(classification_report(ytest_, yhat_test))

            if not save_evaluation_to is None:
                scikitplot.metrics.plot_confusion_matrix(ytest_, yhat_test, figsize=(7,7))
                pyplot.savefig(save_evaluation_to)
        else:
            raise ValueError("Model is not trained yet, call train first")
            
    def predict(self, X, classes=True):
        return (
            np.argmax(self.model.predict(X), axis=1)
            if classes else
            self.model.predict(X)
        )

    def save_model(self, path):
        if self.trained:
            self.model.save(path)
        else:
            raise ValueError("Model is not trained yet, call train first")

    def save_training_history(self, path):
        if self.trained:
            sns.set()
            fig = pyplot.figure(0, (12, 4))

            ax = pyplot.subplot(1, 2, 1)
            sns.lineplot(self.history.epoch, self.history.history['accuracy'], label='train')
            try:
                sns.lineplot(self.history.epoch, self.history.history['val_accuracy'], label='valid')
            except KeyError:
                pass
            pyplot.title('Accuracy')
            pyplot.tight_layout()

            ax = pyplot.subplot(1, 2, 2)
            sns.lineplot(self.history.epoch, self.history.history['loss'], label='train')
            try:
                sns.lineplot(self.history.epoch, self.history.history['val_loss'], label='valid')
            except KeyError:
                pass
            pyplot.title('Loss')
            pyplot.tight_layout()
            pyplot.savefig(path)
        else:
            raise ValueError("Model is not trained yet, call train first")

    def __repr__(self):
        return f"model: {self.__class__.__name__},  trained: {self.trained}"


class DataBuilder():

    def __init__(self, path):
        self.path = path

    def class_image_count(self):
        total_images = 0
        for dir_ in os.listdir(self.path):
            count = 0
            for f in os.listdir(self.path + dir_ + "/"):
                count += 1
            print(f"class {dir_} has {count} images")
            total_images += count
        print(f"total images are {total_images}")

    def build_from_directory(self, img_resize=None):
        img_arr = []
        img_label = []
        label_to_text = {}
        label = 0

        for dir_ in os.listdir(self.path):
            for f in os.listdir(self.path + dir_ + "/"):
                img = cv2.imread(self.path + dir_ + "/" + f, 0)
                if not img_resize is None:
                    img = cv2.resize(img, img_resize)
                img_arr.append(np.expand_dims(img, axis=2))
                img_label.append(label)
            print(f"loaded {dir_} images to numpy arrays...")
            label_to_text[label] = dir_
            label += 1

        img_arr = np.array(img_arr)
        img_label = np.array(img_label)
        img_label = OneHotEncoder(sparse=False).fit_transform(img_label.reshape(-1,1))
        
        return img_arr, img_label, label_to_text


databuilder_obj = DataBuilder("data/")

img_arr, img_label, label_to_text = databuilder_obj.build_from_directory()
img_arr = img_arr / 255.
print(label_to_text)

X_train, X_test, y_train, y_test = train_test_split(img_arr, img_label, shuffle=True, stratify=img_label,
                                                    train_size=0.8, random_state=42)
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape} \n")


lr_schedulers = ReduceLROnPlateau(
    monitor='val_accuracy',
    min_delta=0.0001,
    factor=0.1,
    patience=3,
    min_lr=1e-7,
    verbose=1,
)

callbacks = [lr_schedulers]

train_datagen = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.08,
    height_shift_range=0.1,
    shear_range=0.12,
    zoom_range=0.15,
)

model = CNNModel()
model.train(
    X_train, y_train,
    validation_data = (X_test, y_test),
    batch_size = 32,
    epochs = 16,
    callbacks = callbacks,
    train_datagen=train_datagen
)

model.save_model("dumps/model.h5")
joblib.dump(label_to_text, "dumps/label2text.pkl")

model.evaluate(X_test, y_test, "outputs/confusion_matrix.png")
model.save_training_history("outputs/epoch_metrics.png")
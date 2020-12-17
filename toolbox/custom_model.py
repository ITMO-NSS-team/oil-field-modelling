import numpy as np
from dataclasses import dataclass

import matplotlib.pyplot as plt

plt.style.use("ggplot")

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import models
from keras import optimizers
from keras.layers import Conv2D, MaxPooling2D


class SeismicModel(object):

    def __init__(self,
                 train_path: str,
                 validation_path: str,
                 test_path: str,
                 model_path: str,
                 image_params: tuple = (150, 150, 3)):

        self.model_path = model_path
        self.test_path = test_path
        self.validation_path = validation_path
        self.train_path = train_path
        self.image_params = image_params

        if self.model_path is None:
            self.model = self.get_model()
        else:
            self.model = self.load_model()

        self.train_generator, self.validation_generator, self.test_generator = self._create_data_generators()

    def get_model(self):
        """Function to define the CNN Model"""

        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu',
                                input_shape=self.image_params))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer=optimizers.RMSprop(lr=1e-4),
                      metrics=['acc'])

        return model

    def fit(self,
            steps_per_epoch: int = 100,
            epochs: int = 30):

        results = self.model.fit_generator(
            self.train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=self.validation_generator,
            validation_steps=50)

        return results

    def predict(self):
        self.test_generator.reset()
        predictions = self.model.predict_generator(self.test_generator,
                                                   steps=len(self.test_generator),
                                                   verbose=1)
        filenames = self.test_generator.filenames

        return predictions, filenames

    def load_model(self):
        self.model = load_model(self.model_path)
        return self.model

    def plot_results(self,
                     results,
                     loss_flag: bool = False):

        acc = results.history['acc']
        epochs = range(len(acc))
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.title('Training accuracy')
        plt.legend()
        plt.savefig(r'./accuracy_curve.png', edgecolor='black')

        if loss_flag:
            loss = results.history['loss']
            plt.plot(epochs, loss, 'bo', label='Training loss')
            plt.title('Training loss')
            plt.legend()
            plt.savefig(r'./loss_curve.png', edgecolor='black')

        return

    def _create_data_generators(self):
        data_generator = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True, )

        train_generator = data_generator.flow_from_directory(
            self.train_path,
            target_size=(150, 150),
            batch_size=32,
            class_mode='binary',
            shuffle=True
        )

        validation_generator = data_generator.flow_from_directory(
            self.validation_path,
            target_size=(150, 150),
            batch_size=32,
            class_mode='binary',
            shuffle=True
        )

        test_generator = data_generator.flow_from_directory(
            directory=self.test_path,
            target_size=(150, 150),
            color_mode="rgb",
            batch_size=32,
            class_mode=None,
            shuffle=False)

        return train_generator, validation_generator, test_generator


@dataclass
class OilModel:
    image: np.ndarray
    image_params: tuple = (128, 128, 1)
    kernel_size: str = 3
    model_path: str = None

    def __init__(self,
                 image_params: tuple):
        self.image_params = image_params

        if self.model_path is None:
            self.model = self.get_model()
        else:
            self.model = self.load_model()

    def _conv2d_block(self, input_tensor, n_filters, kernel_size, batchnorm=True):
        """Function to add 2 convolutional layers with the parameters passed to it"""

        # first layer
        x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer='he_normal',
                   padding='same')(input_tensor)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # second layer
        x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer='he_normal',
                   padding='same')(input_tensor)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)

        return x

    def get_model(self, n_filters=16, dropout=0.1, batchnorm=True):
        """Function to define the UNET Model"""

        input_image = self._create_input_tensor()

        # Contracting Path
        c1 = self._conv2d_block(input_tensor=input_image,
                                n_filters=n_filters * 1,
                                kernel_size=self.kernel_size,
                                batchnorm=batchnorm)
        p1 = MaxPooling2D((2, 2))(c1)
        p1 = Dropout(dropout)(p1)

        c2 = self._conv2d_block(input_tensor=p1,
                                n_filters=n_filters * 2,
                                kernel_size=self.kernel_size,
                                batchnorm=batchnorm)
        p2 = MaxPooling2D((2, 2))(c2)
        p2 = Dropout(dropout)(p2)

        c3 = self._conv2d_block(input_tensor=p2,
                                n_filters=n_filters * 4,
                                kernel_size=self.kernel_size,
                                batchnorm=batchnorm)
        p3 = MaxPooling2D((2, 2))(c3)
        p3 = Dropout(dropout)(p3)

        c4 = self._conv2d_block(input_tensor=p3,
                                n_filters=n_filters * 8,
                                kernel_size=self.kernel_size,
                                batchnorm=batchnorm)
        p4 = MaxPooling2D((2, 2))(c4)
        p4 = Dropout(dropout)(p4)

        c5 = self._conv2d_block(input_tensor=p4,
                                n_filters=n_filters * 16,
                                kernel_size=self.kernel_size,
                                batchnorm=batchnorm)

        # Expansive Path
        u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        u6 = Dropout(dropout)(u6)
        c6 = self._conv2d_block(input_tensor=u6,
                                n_filters=n_filters * 8,
                                kernel_size=self.kernel_size,
                                batchnorm=batchnorm)

        u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        u7 = Dropout(dropout)(u7)
        c7 = self._conv2d_block(input_tensor=u7,
                                n_filters=n_filters * 4,
                                kernel_size=self.kernel_size,
                                batchnorm=batchnorm)

        u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        u8 = Dropout(dropout)(u8)
        c8 = self._conv2d_block(input_tensor=u8,
                                n_filters=n_filters * 2,
                                kernel_size=self.kernel_size,
                                batchnorm=batchnorm)

        u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1])
        u9 = Dropout(dropout)(u9)
        c9 = self._conv2d_block(input_tensor=u9,
                                n_filters=n_filters * 1,
                                kernel_size=self.kernel_size,
                                batchnorm=batchnorm)

        outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
        model = Model(inputs=[input_image], outputs=[outputs])

        return model

    def fit(self,
            train_data: tuple,
            val_data: tuple,
            batch_size: int = 3,
            epochs: int = 30,
            callback_list: list = None,
            vis_flag: bool = False):

        self.model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
        # model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])

        if callback_list is None:
            callback_list = [
                EarlyStopping(patience=10, verbose=1),
                ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),
                ModelCheckpoint('model-oil_version2.h5', verbose=1, save_best_only=True, save_weights_only=True)
            ]

        if vis_flag:
            self.model.summary()

        results = self.model.fit(train_data[0],
                                 train_data[1],
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 callbacks=callback_list,
                                 validation_data=val_data)

        return results

    def predict(self, X: np.ndarray):
        return self.model.predict(X, verbose=1)

    def load_model(self):
        return self.model.load_weights(self.model_path)

    def plot_learning_curve(self,
                            data,
                            metric: str):

        plt.figure(figsize=(6, 4))
        plt.title("Learning curve")
        if metric == 'loss':
            plt.plot(data.history["loss"], label="loss")
            plt.plot(data.history["val_loss"], label="val_loss")
            plt.plot(np.argmin(data.history["val_loss"]),
                     np.min(data.history["val_loss"]),
                     marker="x",
                     color="r",
                     label="best model")
            plt.xlabel("Epochs")
            plt.ylabel("log_loss")
            plt.legend()
            plt.savefig('learning curve.png', dpi=300)
        elif metric == 'accuracy':
            plt.figure(figsize=(6, 4))
            plt.title("Learning curve")
            plt.plot(data.history["accuracy"], label="accuracy")
            plt.plot(data.history["val_accuracy"], label="val_accuracy")
            plt.plot(np.argmax(data.history["val_accuracy"]),
                     np.max(data.history["val_accuracy"]),
                     marker="x",
                     color="r",
                     label="best model")
            plt.xlabel("Epochs")
            plt.ylabel("accuracy")
            plt.legend()
            plt.savefig('learning curve acc.png', dpi=300)
        else:
            print('Not supported metric')

    def _create_input_tensor(self):
        input_img = Input(self.image_params, name='img')
        return input_img

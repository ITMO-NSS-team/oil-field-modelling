import os

import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.transform import resize


def get_image(image_params: tuple):
    ids = next(os.walk(r"./images"))[2]  # list of names all images in the given path
    print("No. of images = ", len(ids))

    X = np.zeros((len(ids), image_params[0], image_params[1], 1), dtype=np.float32)
    y = np.zeros((len(ids), image_params[0], image_params[1], 1), dtype=np.float32)

    for n, id_ in enumerate(ids):
        # Load images
        img = load_img(r"./images/" + id_, grayscale=True)
        x_img = img_to_array(img)
        x_img = resize(x_img, (image_params[0], image_params[1], 1), mode='constant', preserve_range=True)
        # Load masks
        mask = img_to_array(load_img(r"./mask/" + id_, grayscale=True))
        mask = resize(mask, (image_params[0], image_params[1], 1), mode='constant', preserve_range=True)
        # Save images
        X[n] = x_img / 255.0
        y[n] = mask / 255.0

    return


def augmentation(X: np.ndarray,
                 y: np.ndarray):
    X = np.append(X, [np.flipud(x) for x in X], axis=0)
    y = np.append(y, [np.flipud(x) for x in y], axis=0)

    # 90 deg rotation
    X = np.append(X, [np.flip(x, axis=(0, 1)) for x in X], axis=0)
    y = np.append(y, [np.flip(x, axis=(0, 1)) for x in y], axis=0)
    return X, y

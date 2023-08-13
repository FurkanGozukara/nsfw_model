import argparse
import json
import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
from os.path import isdir, isfile, join, abspath
from os import listdir

# Constants
INPUT_FOLDER = r"G:\woman dataset\a1_post_processed_processed_clean"  # Set your source folder path here
DESTINATION_FOLDER = r"G:\woman dataset\a1_post_processed_processed_final"  # Set your destination folder path here
CATEGORIES = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']
IMAGE_DIM = 299   # required/default image dimensionality
SAVED_MODEL_PATH="nsfw.299x299.h5"

def load_images(image_paths, image_size, verbose=True):
    '''
    Function for loading images into numpy arrays for passing to model.predict
    inputs:
        image_paths: list of image paths to load
        image_size: size into which images should be resized
        verbose: show all of the image path and sizes loaded
    
    outputs:
        loaded_images: loaded images on which keras model can run predictions
        loaded_image_indexes: paths of images which the function is able to process
    
    '''
    loaded_images = []
    loaded_image_paths = []

    if isdir(image_paths):
        parent = abspath(image_paths)
        image_paths = [join(parent, f) for f in listdir(image_paths) if isfile(join(parent, f))]
    elif isfile(image_paths):
        image_paths = [image_paths]

    for img_path in image_paths:
        try:
            if verbose:
                print(img_path, "size:", image_size)
            image = keras.preprocessing.image.load_img(img_path, target_size=image_size)
            image = keras.preprocessing.image.img_to_array(image)
            image /= 255
            loaded_images.append(image)
            loaded_image_paths.append(img_path)
        except Exception as ex:
            print("Image Load Failure: ", img_path, ex)
    
    return np.asarray(loaded_images), loaded_image_paths

def load_model(model_path):

    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer},compile=False)
    return model


def classify(model, input_paths, image_dim=IMAGE_DIM, predict_args={}):
    """
    Classify given a model, input paths (could be single string), and image dimensionality.
    
    Optionally, pass predict_args that will be passed to tf.keras.Model.predict().
    """
    images, image_paths = load_images(input_paths, (image_dim, image_dim))
    probs = classify_nd(model, images, predict_args)
    return dict(zip(image_paths, probs))


def classify_nd(model, nd_images, predict_args={}):
    """
    Classify given a model, image array (numpy)
    
    Optionally, pass predict_args that will be passed to tf.keras.Model.predict().
    """
    model_preds = model.predict(nd_images, **predict_args)
    # preds = np.argsort(model_preds, axis = 1).tolist()
    
    categories = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']

    probs = []
    for i, single_preds in enumerate(model_preds):
        single_probs = {}
        for j, pred in enumerate(single_preds):
            single_probs[categories[j]] = float(pred)
        probs.append(single_probs)
    return probs

def move_to_classified_folder(image_path, category):
    dest_folder = os.path.join(DESTINATION_FOLDER, category)
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    shutil.move(image_path, dest_folder)

def main():
    if not os.path.exists(INPUT_FOLDER):
        raise ValueError("Source folder does not exist.")

    if not os.path.exists(DESTINATION_FOLDER):
        os.makedirs(DESTINATION_FOLDER)

    model = load_model(SAVED_MODEL_PATH)
    image_preds = classify(model, INPUT_FOLDER, IMAGE_DIM)
    print(json.dumps(image_preds, indent=2), '\n')

    for image_path, preds in image_preds.items():
        max_category = max(preds, key=preds.get)
        move_to_classified_folder(image_path, max_category)


if __name__ == "__main__":
    main()

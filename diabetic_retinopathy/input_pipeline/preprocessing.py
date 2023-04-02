import gin
<<<<<<< HEAD
import tensorflow as tf
import numpy as np
import logging
import tensorflow_addons as tfa
=======
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa

>>>>>>> 3cf6961b390ce177f0ec2cd49d1016e3fd9b56bc

@gin.configurable
def preprocess(image, label, img_height, img_width):

<<<<<<< HEAD
    #Dataset preprocessing: Normalizing and resizing
    
    # Normalize image: `uint8` -> `float32`.
    image = tf.cast(image, tf.float32) / 255.
    
    # Resize image
    image = tf.image.resize(image, size=(img_height, img_width))

=======
    """Dataset preprocessing: Normalizing and resizing

    # Normalize image: `uint8` -> `float32`.
    image = tf.cast(image, tf.float32) / 255.
    
    # Resize image
    image = tf.image.resize(image, size=(img_height, img_height))
    # print(image.numpy())
>>>>>>> 3cf6961b390ce177f0ec2cd49d1016e3fd9b56bc
    return image, label

"""Data augmentation"""
@gin.configurable
def augment(image, label, data_aug_on):
    if data_aug_on:
        # the value used in the below positional arguments are tested to suit this dataset.
        #image = tf.image.resize(tf.image.resize_with_crop_or_pad(image, target_height = np.random.randint(low = 180, high = 230), target_width = np.random.randint(low = 180, high = 230)), size= (img_height, img_width))
        image = tf.image.random_flip_left_right(image)
        #image = tf.image.rot90(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta = 0.1)
        image = tf.image.random_saturation(image, lower = 0.75, upper = 1.2)
        image = tf.image.random_hue(image, max_delta = 0.01)
        image = tf.image.random_contrast(image, lower = 0.75, upper = 1.2)
        random_angles = tf.random.uniform(shape=(), minval= (-np.pi) / 8, maxval= np.pi / 8)
        image = tfa.image.rotate(image, random_angles)
        #image = tf.image.adjust_brightness(image, delta = np.random.uniform(0, 0.20))
        #image = tf.image.adjust_hue(image, delta = 0.2)
        ##image = tf.image.adjust_gamma(image, gamma = 0.5, gain=1)
        #image = tf.image.per_image_standardization(image)
        #image = tf.image.convert_image_dtype(image, dtype=tf.float32, saturate = False)

<<<<<<< HEAD
=======
def augment(image, label):
    """Data augmentation"""
   # image = tf.image.resize(tf.image.resize_with_crop_or_pad(image, target_height = np.random.randint(low = 180, high = 230), target_width = np.random.randint(low = 180, high = 230)), size= (img_height, img_width))
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.adjust_brightness(image, delta = np.random.uniform(0, 0.20))
    image = tf.image.adjust_contrast(image, contrast_factor = np.random.uniform(0.5, 1))

>>>>>>> 3cf6961b390ce177f0ec2cd49d1016e3fd9b56bc
    return image, label

import gin
import os
import pathlib
import glob
import pandas as pd
import numpy as np
import logging
import tensorflow as tf
from sklearn.model_selection import train_test_split
import tensorflow_datasets as tfds
<<<<<<< HEAD
import matplotlib.pyplot as plt
from input_pipeline.preprocessing import preprocess, augment
from collections import Counter
=======

from input_pipeline.preprocessing import preprocess, augment
>>>>>>> 3cf6961b390ce177f0ec2cd49d1016e3fd9b56bc

@gin.configurable
def load(name, data_dir, eyepacs_train_class0, eyepacs_val_class0, eyepacs_test_class0, idrid_oversample_on, graham_preprocessing_on):
    if name == "idrid":
        logging.info(f"Preparing dataset {name}...")
        ds_info = []
        

        '''
        TARGET_MIN_COUNTING = 250
        def get_num_of_repetition_for_class(class_id):
            counting = label_counter[class_id]
            if counting >= TARGET_MIN_COUNTING:
                return 1.0
            num_to_repeat = TARGET_MIN_COUNTING / counting
            return num_to_repeat
        
        numbers_of_repetition_for_classes  = {class_id: get_num_of_repetition_for_class(class_id) for class_id in range(2)}
        '''

        def label_from_csv(data_dir, base_path, graham_preprocessing_on):
            label = pd.read_csv(data_dir)
            if graham_preprocessing_on:
                label["Image path"]  = base_path+"preprocessed_"+label["Image name"]+".jpg" # For IDRID with bt graham
            else:
                label["Image path"]  = base_path+label["Image name"]+".jpg" # For IDRID without bt graham
            return label

        def parse_function(filename, label):

            image_string = tf.io.read_file(filename)

<<<<<<< HEAD
            image = tf.io.decode_image(image_string, channels=3, expand_animations = False)

            #image = tf.image.convert_image_dtype(image, tf.float32) 
=======
            image = tf.io.decode_jpeg(image_string, channels=3)
>>>>>>> 3cf6961b390ce177f0ec2cd49d1016e3fd9b56bc

            return image, label

<<<<<<< HEAD
        def _preprocess(images, label):
            label = 0 if label < 2 else 1
            return images, label

        # preparing training data 
        label_counter = Counter({1: 259, 0: 154}) #class 1 = 259 samples and class 0 = 154 samples
        #running the above commented get_num_of_repetition_for_class will yield numbers_of_repetition_for_classes
        numbers_of_repetition_for_classes = {0: 1.6233766233766234, 1: 1.0} #class 1 more than 250 so return 1, where the class 0 = 250/154 = 1.623. 

        keys_tensor = tf.constant([k for k in numbers_of_repetition_for_classes])
        vals_tensor = tf.constant([numbers_of_repetition_for_classes[k] for k in numbers_of_repetition_for_classes])
        table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor), -1)
        def get_num_of_repetition_for_example(training_example):
            _, label = training_example
            
            num_to_repeat = table.lookup(label)
            num_to_repeat_integral = tf.cast(int(num_to_repeat), tf.float32)
            residue = num_to_repeat - num_to_repeat_integral
            
            num_to_repeat = num_to_repeat_integral + tf.cast(tf.random.uniform(shape=()) <= residue, tf.float32)
            
            return tf.cast(num_to_repeat, tf.int64)

        train_base_path = data_dir + "IDRID_dataset/images/train/"
        final_df = label_from_csv(data_dir + "IDRID_dataset/labels/train.csv", train_base_path, graham_preprocessing_on)
        #template = 'After Imagepath: {}, labels: {}' #to check the data order and to prove it is working uncomment the above 2 lines and these two lines
        #logging.info(template.format(final_df["Image path"][0], final_df["Retinopathy grade"][0]))

        # For train and validation dataset
        dataset = tf.data.Dataset.from_tensor_slices((final_df["Image path"], final_df["Retinopathy grade"]))
        #template = 'After dataset: {}'
        #logging.info(template.format(sum(1 for _ in dataset)))
        dataset = dataset.map(parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
=======
        # preparing training data 
        
        imagepath = glob.glob(data_dir + "/images/train/*jpg")
       
        labels = label_from_csv(data_dir + "/labels/train.csv")
        # print (imagepath[0:2])
        # print (labels[0:2])
        train_image, val_image, train_labels, val_labels = train_test_split(imagepath, labels, test_size = 0.1)
>>>>>>> 3cf6961b390ce177f0ec2cd49d1016e3fd9b56bc
        
        if idrid_oversample_on:
            dataset = dataset.flat_map(lambda image, label: tf.data.Dataset.from_tensors((image, label)).repeat(get_num_of_repetition_for_example((image, label))))
        
        template = 'After dataset: {}'
        logging.info(template.format(sum(1 for _ in dataset)))
        train_size = int(0.9 * sum(1 for _ in dataset)) #finds the length of the dataset and splits
        val_size = int(0.1 * sum(1 for _ in dataset))
        train_ds = dataset.take(train_size)
        train_ds = train_ds.shuffle(train_size)

        val_ds = dataset.skip(train_size)
        val_ds = val_ds.shuffle(val_size)

        # For test dataset
        test_base_path = data_dir + "IDRID_dataset/images/test/"
        final_test_df= label_from_csv(data_dir + "IDRID_dataset/labels/test.csv", test_base_path, graham_preprocessing_on) 

        test_ds = tf.data.Dataset.from_tensor_slices((final_test_df["Image path"], final_test_df["Retinopathy grade"]))
        test_ds = test_ds.map(parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)    
        test_ds = test_ds.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)    
        test_size = sum(1 for _ in test_ds)
        test_ds = test_ds.shuffle(test_size)

        return prepare(train_ds, val_ds, test_ds, ds_info)
        
    elif name == "eyepacs":
        logging.info(f"Preparing dataset {name}...")
        if graham_preprocessing_on:
            (ds_train, ds_val, ds_test), ds_info = tfds.load(
                'diabetic_retinopathy_detection/btgraham-300',
                split=['train', 'validation', 'test'],
                shuffle_files=True,
                with_info=True,
                data_dir=data_dir
            )
        
        else:
            (ds_train, ds_val, ds_test), ds_info = tfds.load(
                'diabetic_retinopathy_detection/250K',
                split=['train', 'validation', 'test'],
                shuffle_files=True,
                with_info=True,
                data_dir=data_dir
            )

        def _preprocess(img_label_dict):
            img_label_dict['label'] = 0 if img_label_dict['label'] < 2 else 1
            return img_label_dict['image'], img_label_dict['label']

        ds_train = ds_train.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_train_0 = ds_train.filter(lambda x, y: y == 0).take(eyepacs_train_class0)
        ds_train_1 = ds_train.filter(lambda x, y: y == 1).take(eyepacs_train_class0)
        ds_train = ds_train_1.concatenate(ds_train_0)
        #ds_train = ds_train.filter(lambda x, y: y == 1).take(train_class0)
        ds_train = ds_train.shuffle(eyepacs_train_class0*2, reshuffle_each_iteration=True)

        ds_val = ds_val.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_val_0 = ds_val.filter(lambda x, y: y == 0).take(eyepacs_val_class0)
        ds_val_1 = ds_val.filter(lambda x, y: y == 1).take(eyepacs_val_class0)
        ds_val = ds_val_1.concatenate(ds_val_0)
        #ds_val = ds_val.filter(lambda x, y: y == 1).take(val_class0)
        ds_val = ds_val.shuffle(eyepacs_val_class0*2, reshuffle_each_iteration=True)
    
        ds_test = ds_test.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_test_0 = ds_test.filter(lambda x, y: y == 0).take(eyepacs_test_class0)
        ds_test_1 = ds_test.filter(lambda x, y: y == 1).take(eyepacs_test_class0)
        ds_test = ds_test_1.concatenate(ds_test_0)
        ds_test = ds_test.shuffle(eyepacs_test_class0*2, reshuffle_each_iteration=True)

        return prepare(ds_train, ds_val, ds_test, ds_info)

    else:
        raise ValueError

@gin.configurable
def prepare(ds_train, ds_val, ds_test, ds_info, batch_size, caching):
    # Prepare training dataset
    ds_train = ds_train.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if caching:
        ds_train = ds_train.cache()
    ds_train = ds_train.map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples // 10)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.repeat(-1)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare validation dataset
    ds_val = ds_val.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if caching:
        ds_val = ds_val.cache()

    #ds_val= ds_val.shuffle(8, reshuffle_each_iteration=True)
    ds_val = ds_val.batch(batch_size)
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare test dataset
    ds_test = ds_test.map(
       preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(batch_size)
    if caching:
        ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
    logging.info(f"Done dataset preparation...")
    return ds_train, ds_val, ds_test, ds_info

def prepare1(ds_train, ds_val, ds_test, ds_info, batch_size=64, caching=False):
    # Prepare training dataset
    ds_train = ds_train.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if caching:
        ds_train = ds_train.cache()
    #ds_train = ds_train.map(
    #    augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples // 10)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.repeat(-1)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare validation dataset
    ds_val = ds_val.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.batch(batch_size)
    if caching:
        ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare test dataset
    ds_test = ds_test.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(batch_size)
    if caching:
        ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_val, ds_test, ds_info

if __name__ == "__main__":
        data_dir = r"C:\Users\johnp\Desktop\John - Studies\University of Stuttgart - New\Deep Learning Lab - 2022\Datasets\idrid\IDRID_dataset"
        load(name = "idrid", data_dir = data_dir)



'''
def prepare(ds_train, ds_val, ds_test, ds_info, train_class0, val_class0, test_class0, batch_size, caching):
    # Prepare training dataset
    ds_train = ds_train.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if caching:
        ds_train = ds_train.cache()
    #ds_train = ds_train.map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples // 10)
    ds_train = ds_train.shuffle(train_class0*2, reshuffle_each_iteration=True)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.repeat(-1)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare validation dataset
    ds_val = ds_val.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if caching:
        ds_val = ds_val.cache()

    ds_val = ds_val.shuffle(val_class0*2, reshuffle_each_iteration=True)
    #ds_val= ds_val.shuffle(8, reshuffle_each_iteration=True)
    ds_val = ds_val.batch(batch_size)
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare test dataset
    ds_test = ds_test.map(
       preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.shuffle(test_class0*2, reshuffle_each_iteration=True)
    ds_test = ds_test.batch(batch_size)
    if caching:
        ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
    logging.info(f"Done dataset preparation...")
    return ds_train, ds_val, ds_test, ds_info
'''
import logging
import tensorflow as tf
import gin
import numpy as np

def set_loggers(path_log=None, logging_level=0, b_stream=False, b_debug=False):

    # std. logger
    logger = logging.getLogger()
    logger.setLevel(logging_level)

    # tf logger
    logger_tf = tf.get_logger()
    logger_tf.setLevel(logging_level)

    if path_log:
        file_handler = logging.FileHandler(path_log)
        logger.addHandler(file_handler)
        logger_tf.addHandler(file_handler)

    # plot to console
    if b_stream:
        stream_handler = logging.StreamHandler()
        logger.addHandler(stream_handler)

    if b_debug:
        tf.debugging.set_log_device_placement(False)

def Callback_EarlyStopping(LossList, min_delta=0.01, patience=2):
    #No early stopping for 2*patience epochs 
    if len(LossList)//patience < 2 :
        return False

    #Mean loss for last patience epochs and second-last patience epochs
    mean_previous = np.mean(LossList[::-1][patience:2*patience]) #second-last
    mean_recent = np.mean(LossList[::-1][:patience]) #last

    #you can use relative or absolute change
    delta_abs = np.abs(mean_recent - mean_previous) #abs change
    delta_abs = np.abs(delta_abs / mean_previous)  #relative change
    if delta_abs < min_delta :
        #logging.info("Loss didn't change much from last %d epochs"%(patience))
        #logging.info("Percent change in loss value:", delta_abs*1e2)
        return True
    else:
        return False
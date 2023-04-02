import gin
import tensorflow as tf
import matplotlib.cm as cm
import numpy as np
import logging


# To generate the Gradcam heatmap and impose that on the image 
def grad_cam(img, model, last_layer_name, class_index=None):
    """
    Parameters:
    img = Already preprocessed image

    """
    
    #TO DO ADD THE PREPROCESSING STEP IN HERE AND THEN FEED THE IMAGE FURTHER

    # Remove last layer's softmax or binary decision activation layer.
    class_belong = model(img, training=False)
    logging.info(class_belong)
    model.layers[-1].activation = None

    # model to map the input image to the activations of the last feature layer as well as output pred layer
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_layer_name).output, model.output]
    )

    # Now, we compute the gradient of the top predicted class for our input image w. respect to the activations of the last feature layer
    with tf.GradientTape() as tape:
        last_layer_output, preds = grad_model(img)
        #e.g) assume last_layer_output.shape = (1, 10, 10, 2048) - Note not a original dim
        #preds.shape (1, 2)
        if class_index is None:
            class_index = tf.argmax(preds[0])
        class_channel = preds[0, class_index]
        
        #class_channel contains the output of the final prediction layer w/o softmax precentage value. 

    #The gradient of the output neuron (top predicted) with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_layer_output)
    #grads.shape same as last_layer_output.shape but contains gradients.

    #Take an average over the height and width of the final feature layer, results in a vector over a specific feature map channel
    avg_grads = tf.reduce_mean(grads, axis = (0,1,2))
    #avg_grads.shape = (1, no_of_channels) e.g (1, 2048)

    last_layer_output = last_layer_output[0]
    #last_layer_output = last_layer_output

    #last_layer_output.shape = (10,10,2048)- Note not a original dim
    heatmap = last_layer_output @ avg_grads[..., tf.newaxis]
    #heatmap.shape = (10, 10, 1) dot producted over the no.of channels - Note not a original dim

    heatmap = tf.squeeze(heatmap) #to remove over a dim resulting in (10, 10)

    # For visualization purpose, normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap

@gin.configurable
def superimpose_gradcam(heatmap, img_path, alpha=0.4):
    
    """
    Parameters:
    img_path = Without preprocessing the image
    """
    
    # Load the original image
    img = tf.keras.preprocessing.image.load_img(img_path)
    img = tf.keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)
    # heatmap.shape = (10, 10, 1) e.g from before

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    #jet_colors.shape = (256, 3)
    jet_heatmap = jet_colors[heatmap]
    #jet_heatmap.shape = (10, 10, 3)

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap) #convert jet_heatmap array to image
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0])) #resize jet_heatmap image to original image width and height
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap) #convert jet_heatmap image to array again

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    #superimposed_img.save(cam_path)

    return superimposed_img


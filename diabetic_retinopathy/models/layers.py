import gin
import tensorflow as tf
import matplotlib.cm as cm
import numpy as np

<<<<<<< HEAD
#### CUSTOM VGG MODEL - LAYERS
=======

>>>>>>> 3cf6961b390ce177f0ec2cd49d1016e3fd9b56bc
@gin.configurable
def vgg_block(inputs, filters, kernel_size):
    """A single VGG block consisting of two convolutional layers, followed by a max-pooling layer.

    Parameters:
        inputs (Tensor): input of the VGG block
        filters (int): number of filters used for the convolutional layers
        kernel_size (tuple: 2): kernel size used for the convolutional layers, e.g. (3, 3)

    Returns:
        (Tensor): output of the VGG block
    """

    out = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', activation=tf.nn.relu)(inputs)
    out = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', activation=tf.nn.relu)(out)
    out = tf.keras.layers.MaxPool2D((2, 2))(out)

    return out

<<<<<<< HEAD
#### TRANSFORMER MODEL - LAYERS
#To convert the input image into small patches for further patchencoding
class Patches_Creator(tf.keras.layers.Layer): 
    def __init__(self, patch_size):
        super(Patches_Creator, self).__init__()
        self.patch_size = patch_size #the size of the each patches (8) of a image 
    
    def call(self, img):
        # image shape would be 4D e.g [batch size,  height, width, channel][1, 256, 256, 3]
        batch_size = tf.shape(img)[0] 

        #to extract the patches, note the stride and sizes has the same to not miss any pixel. 
        patch = tf.image.extract_patches(
            images=img,
            sizes=[1, self.patch_size, self.patch_size, 1], 
            strides=[1, self.patch_size, self.patch_size, 1], 
            rates=[1, 1, 1, 1],
            padding="VALID",
            )

        #patch.shape = [1, 32, 32, 192] how? [1, 256/patch_size=8, 256/patch_size=8, 8*8*3]
        patch_dim = patch.shape[-1] #elements per patch 
        patch = tf.reshape(patch, [batch_size, -1, patch_dim]) # =1 will add the dimentions together
        #patch.shape = [batch_size, 1024, 192] how? [1, 32*32, 192]
        return patch

#To encode the patches by projecting it into a vector of size proj_dim (our case 128). Input shape [batch_size, 1024, 192] -> output shape [batch_size, 1024, proj_dim=128]
class Patches_Encoder(tf.keras.layers.Layer):
    def __init__(self, no_of_patches, proj_dim):
        super(Patches_Encoder, self).__init__()
        self.no_of_patches = no_of_patches
        self.proj = tf.keras.layers.Dense(units=proj_dim)
        #self.proj.shape = [batch_size, 1024, proj_dim=128]
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=no_of_patches, output_dim=proj_dim
        ) #embedding layer create the uniform distributed (by default) weight which are added later to the patch values
        #self.position_embedding.shape = [1024, proj_dim=128]

    def call(self, patch):
        positions = tf.range(start=0, limit=self.no_of_patches, delta=1) #to generate a tensor with start 0, till no_of_patches, step 1. 
        #positions.shape = [1024, ]
        encoded = self.proj(patch) + self.position_embedding(positions) #patch's value is added with the weights with positions variable as index
        #encoded.shape = [batch_size, 1024, 128]
        return encoded


def multilayer_preceptron(x, no_of_hidden_units, dropout_rate):
    for hidden_units in no_of_hidden_units:
        x = tf.keras.layers.Dense(hidden_units, activation=tf.nn.gelu)(x)
        x = tf.keras.layers.Dropout(rate = dropout_rate)(x)
    return x

def transformer_unit(input_encoded_patches, no_of_heads, proj_dim, transformer_units, dropout_rate):
    #LayerNormalization - Normalize the activations of the previous layer for each given example in a batch independently, rather than across a batch like Batch Normalization..
    x1 = tf.keras.layers.BatchNormalization(epsilon=1e-6)(input_encoded_patches)
    #x1.shape = [batch_size, 1024, 128]

    #To create a multi-head attention layer
    attention_out = tf.keras.layers.MultiHeadAttention(
        num_heads=no_of_heads, key_dim = proj_dim, dropout=dropout_rate
        )(x1, x1) #Query and Key (dim d=128) are the same i.e., encoded patches. key_dim = size of each attention head for query and key.
    #attention_out.shape = [batch_size, 1024, 128]

    #SKIP CONNECTION 1
    #Add and LayerNormalization as per the attention is all you need paper. 
    x2 = tf.keras.layers.Add()([attention_out, input_encoded_patches])
    #x2.shape = [batch_size, 1024, 128]

    #LayerNormalization - Normalize the activations of the previous layer for each given example in a batch independently, rather than across a batch like Batch Normalization..
    x3 = tf.keras.layers.BatchNormalization(epsilon=1e-6)(x2)
    #x3.shape = [batch_size, 1024, 128]

    #one feed forward layer - Use MLP layer with proj_dim as the inputs
    x4 = multilayer_preceptron(x3, no_of_hidden_units = transformer_units, dropout_rate=dropout_rate) #try changing the 0.1 value for dropouts.

    #SKIP CONNECTION 2
    input_encoded_patches = tf.keras.layers.Add()([x4, x2])

    return input_encoded_patches

#### RESNET CUSTOM MODEL - LAYERS
=======
>>>>>>> 3cf6961b390ce177f0ec2cd49d1016e3fd9b56bc
@gin.configurable
def identityblock(x, kernel_size, filters):
    # Layer 1
    fx = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')(x)
    fx = tf.keras.layers.BatchNormalization(axis=3)(fx)
    fx = tf.keras.layers.ReLU()(fx)
    # Layer 2
    fx = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')(fx)
    out = tf.keras.layers.BatchNormalization()(fx)
    # Add Residue
    out = tf.keras.layers.Add()([x,fx])
    out = tf.keras.layers.ReLU()(out)
    
    return out

@gin.configurable
def convblock(x, kernelsize, filters):
    # skip connection
    x_skip = x
    # Layer 1
    x = tf.keras.layers.Conv2D(filters, kernelsize, padding = 'same', strides = (2,2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Layer 2
    x = tf.keras.layers.Conv2D(filters, kernelsize, padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # Processing Residue with conv(1,1)
    x_skip = tf.keras.layers.Conv2D(filters, (1,1), strides = (2,2))(x_skip)
    # Add Residue
    x = tf.keras.layers.Add()([x, x_skip])     
    x = tf.keras.layers.Activation('relu')(x)
<<<<<<< HEAD
    return x

=======
    return x
>>>>>>> 3cf6961b390ce177f0ec2cd49d1016e3fd9b56bc

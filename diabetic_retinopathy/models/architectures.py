import gin
import tensorflow as tf
<<<<<<< HEAD
from models.layers import vgg_block
from models.layers import Patches_Creator, Patches_Encoder, multilayer_preceptron, transformer_unit
from models.layers import identityblock, convblock

#### STANDARD PRETRAINED MODEL
@gin.configurable
def standard_models(input_shape, dropout_rate, n_classes, model_name):
    #model = tf.keras.applications.ResNet101(weights="imagenet", include_top=True, input_tensor=tf.keras.layers.Input(shape=input_shape))
    if model_name == "resnet50":
        model = tf.keras.applications.ResNet50(weights="imagenet", include_top=True, input_tensor=tf.keras.layers.Input(shape=input_shape))
    elif model_name == "inceptionV3":
        model = tf.keras.applications.InceptionV3(weights="imagenet", include_top=True, input_tensor=tf.keras.layers.Input(shape=input_shape))
    elif model_name == "xception":
        model = tf.keras.applications.Xception(weights="imagenet", include_top=True, input_tensor=tf.keras.layers.Input(shape=input_shape))
    x = tf.keras.layers.Dense(512, activation='relu')(model.layers[-2].output)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    out = tf.keras.layers.Dense(n_classes, activation='sigmoid')(x)
    return tf.keras.Model(inputs=model.input, outputs=out, name=model_name)
=======

from models.layers import vgg_block, identityblock, convblock
>>>>>>> 3cf6961b390ce177f0ec2cd49d1016e3fd9b56bc

#### ENSEMBLE MODEL
@gin.configurable
def ensemble_model(input_shape, standard_models):
    model_1 = standard_models(model_name="resnet50")
    model_2 = standard_models(model_name="inceptionV3")
    model_3 = standard_models(model_name="xception")
    models = [model_1, model_2, model_3]
    model_input = tf.keras.Input(input_shape)
    model_outputs = [model(model_input) for model in models]
    ensemble_output = tf.keras.layers.Average()(model_outputs)
    ensemble_model = tf.keras.Model(inputs=model_input, outputs=ensemble_output)
    return ensemble_model

#### VGG CUSTOM MODEL
@gin.configurable
def vgg_like(input_shape, n_classes, base_filters, n_blocks, dense_units, dropout_rate):
    """Defines a VGG-like architecture.

    Parameters:
        input_shape (tuple: 3): input shape of the neural network
        n_classes (int): number of classes, corresponding to the number of output neurons
        base_filters (int): number of base filters, which are doubled for every VGG block
        n_blocks (int): number of VGG blocks
        dense_units (int): number of dense units
        dropout_rate (float): dropout rate

        (keras.Model): keras model object
    
     """
    assert n_blocks > 0, 'Number of blocks has to be at least 1.'

    inputs = tf.keras.Input(input_shape)

    out = vgg_block(inputs, base_filters)
    for i in range(2, n_blocks):
        out = vgg_block(out, base_filters * 2 ** (i))
    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = tf.keras.layers.Dense(dense_units, activation=tf.nn.relu)(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    outputs = tf.keras.layers.Dense(n_classes, activation='sigmoid')(out)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='vgg_like')

#### VISION TRANSFORMER
'''
#https://arxiv.org/abs/1706.03762 - Attention is all you need.
#https://theaisummer.com/vision-transformer/ - vision transformer picture is shown
'''
'''
@gin.configurable
def transformer_backbone(input_sh, patch_size, no_of_patches, proj_dim, transformer_layers, no_of_heads, transformer_units, dropout_rate, data_augment = True):
    inputs = tf.keras.Input(shape = input_sh)
    #inputs.shape = [batch_size, 256, 256, 3]

    #To create the patches from a given image
    patches = Patches_Creator(patch_size)(inputs)
    #patches.shape = [batch_size, 1024, 192]

    #To encode the patched with the embedding weights with the proj_dim
    encoded_patches = Patches_Encoder(no_of_patches, proj_dim)(patches)
    #encoded.shape = [batch_size, 1024, 128]

    for transformer in range(transformer_layers):
        encoded_patches = transformer_unit(encoded_patches, no_of_heads, proj_dim, transformer_units, dropout_rate)
        #encoded_patches.shape = [batch_size, 1024, 128]

    #Final dense layers
    out_representation = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    
    out_representation = tf.keras.layers.GlobalAveragePooling1D()(out_representation)
    #GlobalAveragePooling -> out_representation.shape [batch_size, 128]

    #out_representation = tf.keras.layers.Flatten()(out_representation)
    #Flatten layer -> out_representation.shape [batch_size, 131072]

    # To create the Keras model.
    model = tf.keras.Model(inputs=inputs, outputs=out_representation, name = "transformer_backbone")

    return model


@gin.configurable
def head(head_input_shape, mlp_head_units, dropout_rate, no_of_classes):

    head_inputs = tf.keras.Input(shape = head_input_shape)
    #inputs.shape = [batch_size, 128]

    out_representation = tf.keras.layers.Dropout(dropout_rate)(head_inputs)

    # Add MLP.
    features = multilayer_preceptron(out_representation, no_of_hidden_units=mlp_head_units, dropout_rate=dropout_rate)
    
    # To classify outputs.
    output = tf.keras.layers.Dense(no_of_classes, activation='sigmoid')(features)

    # To create the Keras model.
    model = tf.keras.Model(inputs=head_inputs, outputs=output, name = "head")

    return model



@gin.configurable
def ViT_classifier(input_shape, transformer_backbone, head, transformer_backbone_trainable, MLP_head_trainable):
    ViT_inputs = tf.keras.Input(shape = input_shape)
    #inputs.shape = [batch_size, 256, 256, 3]

    x = transformer_backbone()(ViT_inputs)
    final_out = head()(x)
    
    head.trainable = MLP_head_trainable

    transformer_backbone.trainable = transformer_backbone_trainable
    
    # To create the Keras model.
    model = tf.keras.Model(inputs=ViT_inputs, outputs=final_out, name = "ViT_classifier")

    return model
'''

@gin.configurable
def ViT_Classifier_Viz_Only(input_sh, patch_size, no_of_patches, proj_dim, transformer_layers, no_of_heads, transformer_units, head_input_shape, mlp_head_units, dropout_rate, no_of_classes):
    inputs = tf.keras.Input(shape = input_sh)
    #inputs.shape = [batch_size, 256, 256, 3]

    #To create the patches from a given image
    patches = Patches_Creator(patch_size)(inputs)
    #patches.shape = [batch_size, 1024, 192]

    #To encode the patched with the embedding weights with the proj_dim
    encoded_patches = Patches_Encoder(no_of_patches, proj_dim)(patches)
    #encoded.shape = [batch_size, 1024, 128]

    for transformer in range(transformer_layers):
        encoded_patches = transformer_unit(encoded_patches, no_of_heads, proj_dim, transformer_units, dropout_rate)
        #encoded_patches.shape = [batch_size, 1024, 128]

    #Final dense layers
    out_representation = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    
    out_representation = tf.keras.layers.GlobalAveragePooling1D()(out_representation)
    #GlobalAveragePooling -> out_representation.shape [batch_size, 128]

    #out_representation = tf.keras.layers.Flatten()(out_representation)
    #Flatten layer -> out_representation.shape [batch_size, 131072]
    out_representation = tf.keras.layers.Dropout(dropout_rate)(out_representation)

    # Add MLP.
    features = multilayer_preceptron(out_representation, no_of_hidden_units=mlp_head_units, dropout_rate=dropout_rate)
    
    # To classify outputs.
    output = tf.keras.layers.Dense(no_of_classes, activation='sigmoid')(features)

    # To create the Keras model.
    model = tf.keras.Model(inputs=inputs, outputs=output, name = "ViT_Classifier_Spoof")
    return model


#### CUSTOM RESNET MODEL
@gin.configurable
def resnet_like(input_shape, kernel_size, resblocks, filter_size,num_classes, dense_units1, dense_units2, dropout_rates):
  inputs = tf.keras.Input(shape=input_shape)
  x1 = tf.keras.layers.ZeroPadding2D((3, 3))(inputs)
  x1 = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same')(x1)
  x1 = tf.keras.layers.BatchNormalization()(x1)
  x1 = tf.keras.layers.Activation('relu')(x1)
  x1 = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x1)

  # The Resnet Blocks
  for i in range(len(resblocks)):
        if i == 0:
            # For sub-block 1 Residual/Convolutional block not needed
            for j in range(resblocks[i]):
                x1 = identityblock(x1, kernel_size, filter_size)
        else:
            # One Residual/Convolutional Block followed by Identity blocks
            filter_size = filter_size*2
            x1 = convblock(x1, kernel_size, filter_size)
            for j in range(resblocks[i] - 1):
                x1= identityblock(x1, kernel_size, filter_size)
    
  output = tf.keras.layers.GlobalAveragePooling2D()(x1)
  inp = tf.keras.layers.Dense(dense_units1, activation="relu")(output)
  out = tf.keras.layers.Dropout(dropout_rates)(inp)
  out = tf.keras.layers.Dense(dense_units2, activation="relu")(out)
  out = tf.keras.layers.Dropout(dropout_rates)(out)
  outputs = tf.keras.layers.Dense(num_classes, activation="sigmoid")(out)

  return tf.keras.Model(inputs=inputs, outputs=outputs, name='resnet_like')



<<<<<<< HEAD
=======
    return tf.keras.Model(inputs=inputs, outputs=outputs, name='vgg_like')

@gin.configurable
def backbone(kernel_size, resblocks, filter_size):

  """Defines a customised ResNet based backbone for Transfer Learning, a.k.a feature extractor.
  Parameters:
      input_channels : Number of channels in the input image.
      kernel_size : Size of the kernels in the residual blocks. 
      resblocks : List containing the number of ResNet blocks.
      filter_size : size of the filters for the ResNet blocks (base filter size)
  Returns:
      (keras.Model): keras model object
  """
  # Input layers
  inputs = tf.keras.Input(shape=(None, None, 3))
  x = tf.keras.layers.ZeroPadding2D((3, 3))(inputs)
  x1 = tf.keras.layers.Conv2D(64, kernel_size=(7,7), strides=2, padding='same')(x)
  x1 = tf.keras.layers.BatchNormalization()(x1)
  x1 = tf.keras.layers.Activation('relu')(x1)
  x1 = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x1)

  # The Resnet Blocks
  for i in range(len(resblocks)):
        if i == 0:
            # For sub-block 1 Residual/Convolutional block not needed
            for j in range(resblocks[i]):
                x1 = identityblock(x1, kernel_size, filter_size)
        else:
            # One Residual/Convolutional Block followed by Identity blocks
            filter_size = filter_size*2
            x1 = convblock(x1, kernel_size, filter_size)
            for j in range(resblocks[i] - 1):
                x1= identityblock(x1, kernel_size, filter_size)
    
  outputs = tf.keras.layers.GlobalAveragePooling2D()(x1)

  return tf.keras.Model(inputs=x, outputs=outputs, name='backbone')

@gin.configurable
def Head( num_classes, dense_units1, dense_units2, dropout_rate):

  """Defines the classification head for transfer learning.
  Parameters:
      num_classes  : Number of classes 
      dense_units1 : Number of neurons in the FC layer1
      dense_units2 : Number of neurons in the FC layer2
      dropout_rate : Droupout rate for FC layer

  Returns:
      (keras.Model): keras model object
  """
  inputs = tf.keras.Input(shape=(None,512))
  inp = tf.keras.layers.Dense(dense_units1, activation="relu")(input)
  out = tf.keras.layers.Dropout(dropout_rate)(inp)
  out = tf.keras.layers.Dense(dense_units2, activation="relu")(out)
  out = tf.keras.layers.Dropout(dropout_rate)(out)
  outputs = tf.keras.layers.Dense(num_classes, activation=None)(out)

  return tf.keras.Model(inputs=inp, outputs=outputs, name='Head')

@gin.configurable
def full_model(input_channels, backbone, HEAD, trainable_backbone, trainable_head):
  """Defines a customised full model based backbone and classifiaction head.
  Parameters:
      input_channels : Number of channels in the input image.
      backbone : backmodel model. 
      HEAD : classifaction head model.
      trainable_backbone : False during transfer learning
      trainable_head : always True
  Returns:
      (keras.Model): keras model object
  """
  
  input = tf.keras.Input(shape=(None, None, input_channels))
  
  b = backbone(input)
  pred = HEAD(b)
  HEAD.trainable = trainable_head
  backbone.trainable = trainable_backbone

  return tf.keras.Model(inputs=input, outputs=pred, name='full_model')
@gin.configurable
def new_model(kernel_size, resblocks, filter_size,num_classes, dense_units1, dense_units2, dropout_rate ):
  inputs = tf.keras.Input(shape=(None, None, 3))
  x1 = tf.keras.layers.ZeroPadding2D((3, 3))(inputs)
  x1 = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same')(x1)
  x1 = tf.keras.layers.BatchNormalization()(x1)
  x1 = tf.keras.layers.Activation('relu')(x1)
  x1 = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x1)

  # The Resnet Blocks
  for i in range(len(resblocks)):
        if i == 0:
            # For sub-block 1 Residual/Convolutional block not needed
            for j in range(resblocks[i]):
                x1 = identityblock(x1, kernel_size, filter_size)
        else:
            # One Residual/Convolutional Block followed by Identity blocks
            filter_size = filter_size*2
            x1 = convblock(x1, kernel_size, filter_size)
            for j in range(resblocks[i] - 1):
                x1= identityblock(x1, kernel_size, filter_size)
    
  output = tf.keras.layers.GlobalAveragePooling2D()(x1)
  inp = tf.keras.layers.Dense(dense_units1, activation="relu")(output)
  out = tf.keras.layers.Dropout(dropout_rate)(inp)
  out = tf.keras.layers.Dense(dense_units2, activation="relu")(out)
  out = tf.keras.layers.Dropout(dropout_rate)(out)
  outputs = tf.keras.layers.Dense(num_classes, activation=None)(out)

  return tf.keras.Model(inputs=inputs, outputs=outputs, name='new_model')




if __name__ == "__main__":
  inp = tf.ones([64,256,256, 3], tf.int32)
  backbone_model = backbone(input_channels=3, kernel_size=(3,3))
  pred = backbone_model(inp)
  Head_model = Head(num_classes=2, dense_units1=128, dense_units2=64, dropout_rate=0.2)
  print(Head_model(pred))
  # print(pred)
>>>>>>> 3cf6961b390ce177f0ec2cd49d1016e3fd9b56bc

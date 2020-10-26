import keras
from keras.models import *
from keras.layers import *
from keras import layers

# Source:
# https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py


from .config import IMAGE_ORDERING


if IMAGE_ORDERING == 'channels_first':
    pretrained_url = "https://github.com/fchollet/deep-learning-models/" \
                     "releases/download/v0.2/" \
                     "resnet50_weights_th_dim_ordering_th_kernels_notop.h5"
elif IMAGE_ORDERING == 'channels_last':
    pretrained_url = "https://github.com/fchollet/deep-learning-models/" \
                     "releases/download/v0.2/" \
                     "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"


def one_side_pad(x):
    x = ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING)(x)
    if IMAGE_ORDERING == 'channels_first':
        x = Lambda(lambda x: x[:, :, :-1, :-1])(x)
    elif IMAGE_ORDERING == 'channels_last':
        x = Lambda(lambda x: x[:, :-1, :-1, :])(x)
    return x


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at
                     main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters

    if IMAGE_ORDERING == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x1 = Conv2D(filters1, (1, 1), data_format=IMAGE_ORDERING,
               name=conv_name_base + '2a')(input_tensor)
    x1 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x1)
    x1 = Activation('relu')(x1)
   # x1 = concatenate([x1, input_tensor], axis = -1)
    
    x2 = Conv2D(filters2, kernel_size, data_format=IMAGE_ORDERING,
               padding='same', name=conv_name_base + '2b')(x1)
    x2 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x2)
    x2 = Activation('relu')(x2)
    x2 = concatenate([x2, x1], axis = -1)
    
    x3 = Conv2D(filters3, (1, 1), data_format=IMAGE_ORDERING,
               name=conv_name_base + '2c')(x2)
    x3 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x3)
    x3 = layers.add([x3, input_tensor])
    x3 = Activation('relu')(x3)
    return x3


def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(1,1)):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at
                     main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with
    strides=(2,2) and the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters

    if IMAGE_ORDERING == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x1 = Conv2D(filters1, (1, 1), data_format=IMAGE_ORDERING, strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x1 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x1)
    x1 = Activation('relu')(x1)
    x1 = concatenate([x1, input_tensor], axis = -1)
    
    x2 = Conv2D(filters2, kernel_size, data_format=IMAGE_ORDERING,
               padding='same', name=conv_name_base + '2b')(x1)
    x2 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x2)
    x2 = Activation('relu')(x2)
    x2 = concatenate([x2, x1, input_tensor], axis = -1)
    
    x3 = Conv2D(filters3, (1, 1), data_format=IMAGE_ORDERING,
               name=conv_name_base + '2c')(x2)
    x3 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x3)

    shortcut = Conv2D(filters3, (1, 1), data_format=IMAGE_ORDERING,
                      strides=strides, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x3, shortcut])
    x = Activation('relu')(x)
    return x
   # return x3


def get_resnet50_encoder(input_height=224,  input_width=224, channels=4,
                         pretrained='None',
                         include_top=True, weights='None',
                         input_tensor=None, input_shape=None,
                         pooling=None,
                         classes=1000):

    assert input_height % 32 == 0
    assert input_width % 32 == 0

    if IMAGE_ORDERING == 'channels_first':
        img_input = Input(shape=(channels, input_height, input_width))
    elif IMAGE_ORDERING == 'channels_last':
        img_input = Input(shape=(input_height, input_width, channels))

    if IMAGE_ORDERING == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x1 = ZeroPadding2D((3, 3), data_format=IMAGE_ORDERING)(img_input)
    x1 = Conv2D(64, (7, 7), data_format=IMAGE_ORDERING,
               strides=(2, 2), name='conv1')(x1)
    f1 = x1

    x2 = BatchNormalization(axis=bn_axis, name='bn_conv1')(x1)
    x2 = Activation('relu')(x2)
    x2 = MaxPooling2D((3, 3), data_format=IMAGE_ORDERING, strides=(2, 2))(x2)
    
   # x2 = MaxPooling2D((3, 3), data_format=IMAGE_ORDERING, strides=(2, 2))(x2)
    x2_1 = conv_block(x2, 3, [64, 64, 128], stage=2, block='a', strides=(1, 1))
  #  x2_1 = concatenate([x2_1, x2], axis = -1)
    x2_2 = identity_block(x2_1, 3, [64, 64, 128], stage=2, block='b')
 #   x2_2 = concatenate([x2_2, x2_1, x2], axis = -1)
    x2_3 = identity_block(x2_2, 3, [64, 64, 128], stage=2, block='c')
    x2_3 = concatenate([x2_3, x2_2, x2_1, x2], axis = -1)
    f2 = one_side_pad(x2_3)
    
    x3_1 = MaxPooling2D((3, 3), data_format=IMAGE_ORDERING, strides=(2, 2), padding = 'same')(x2_3)
    x3_2 = conv_block(x3_1, 3, [64, 64, 128], stage=3, block='a')
#    x3_2 = concatenate([x3_2, x3_1], axis = -1)
    x3_3 = identity_block(x3_2, 3, [64, 64, 128], stage=3, block='b')
#    x3_3 = concatenate([x3_3, x3_2, x3_1], axis = -1)
    x3_4 = identity_block(x3_3, 3, [64, 64, 128], stage=3, block='c')
#    x3_4 = concatenate([x3_4, x3_3, x3_2, x3_1], axis = -1)
    x3_5 = identity_block(x3_4, 3, [64, 64, 128], stage=3, block='d')
    x3_5 = concatenate([x3_5, x3_4, x3_3, x3_2, x3_1], axis = -1)
    f3 = x3_5
    
    x4_1 = MaxPooling2D((3, 3), data_format=IMAGE_ORDERING, strides=(2, 2), padding = 'same')(x3_5)
    x4_2 = conv_block(x4_1, 3, [128, 128, 512], stage=4, block='a')
#    x4_2 = concatenate([x4_2, x4_1], axis = -1)
    x4_3 = identity_block(x4_2, 3, [128, 128, 512], stage=4, block='b')
#    x4_3 = concatenate([x4_3, x4_2, x4_1], axis = -1)
    x4_4 = identity_block(x4_3, 3, [128, 128, 512], stage=4, block='c')
#    x4_4 = concatenate([x4_4, x4_3, x4_2, x4_1], axis = -1)
    x4_5 = identity_block(x4_4, 3, [128, 128, 512], stage=4, block='d')
#    x4_5 = concatenate([x4_5, x4_4, x4_3, x4_2, x4_1], axis = -1)
    x4_6 = identity_block(x4_5, 3, [128, 128, 512], stage=4, block='e')
#    x4_6 = concatenate([x4_6, x4_5, x4_4, x4_3, x4_2, x4_1], axis = -1)
    x4_7 = identity_block(x4_6, 3, [128, 128, 512], stage=4, block='f')
    x4_7 = concatenate([x4_7, x4_6, x4_5, x4_4, x4_3, x4_2, x4_1], axis = -1)
    f4 = x4_7
    
    x5_1 = MaxPooling2D((3, 3), data_format=IMAGE_ORDERING, strides=(2, 2), padding = 'same')(x4_7)
    x5_2 = conv_block(x5_1, 3, [256, 256, 512], stage=5, block='a')
 #   x5_2 = concatenate([x5_2, x5_1], axis = -1)
    x5_3 = identity_block(x5_2, 3, [256, 256, 512], stage=5, block='b')
 #   x5_3 = concatenate([x5_3, x5_2, x5_1], axis = -1)
    x5_4 = identity_block(x5_3, 3, [256, 256, 512], stage=5, block='c')
    x5_4 = concatenate([x5_4, x5_3, x5_2, x5_1], axis = -1)
    f5 = x5_4

    x6 = AveragePooling2D(
        (7, 7), data_format=IMAGE_ORDERING, name='avg_pool')(x5_4)
    # f6 = x

    if pretrained == 'imagenet':
        weights_path = keras.utils.get_file(
            pretrained_url.split("/")[-1], pretrained_url)
        Model(img_input, x).load_weights(weights_path)

    return img_input, [f1, f2, f3, f4, f5]

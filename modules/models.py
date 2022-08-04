from absl import flags
from absl.flags import FLAGS
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import (
    Add,
    Dropout,
    Conv2D,
    Dense,
    Flatten,
    GlobalAvgPool2D,
    Lambda,
    LeakyReLU,
    BatchNormalization,
    ZeroPadding2D,
)
from tensorflow.keras import Model, Input
from tensorflow.keras.losses import binary_crossentropy,sparse_categorical_crossentropy


# function-like definitions
def resblock_base(x,num_fil):
    pre = x
    x = ZeroPadding2D(padding=1,data_format='channels_last')(x)
    x = Conv2D(filters=num_fil,kernel_size=(3,3))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = ZeroPadding2D(padding=1, data_format='channels_last')(x)
    x = Conv2D(filters=num_fil,kernel_size=(3,3))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Add()([pre,x])
    return x


def resblock_conv(x,num_fil):
    pre = x
    x = ZeroPadding2D(padding=1, data_format='channels_last')(x)
    x = Conv2D(filters=num_fil,kernel_size=(3,3),strides=2)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = ZeroPadding2D(padding=1, data_format='channels_last')(x)
    x = Conv2D(filters=num_fil,kernel_size=(3,3),strides=1)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    pre = Conv2D(filters=num_fil, kernel_size=(1, 1),strides=2)(pre)
    pre = BatchNormalization()(pre)
    pre = LeakyReLU(alpha=0.1)(pre)
    x = Add()([pre,x])
    return x


def Resnet18():
    def bankbone(x_input):
        x = inputs = Input(shape=x_input.shape[1:])
        x = Conv2D(filters=64,kernel_size=(1,1))(x)
        x = resblock_base(x, num_fil=64)
        x = resblock_base(x, num_fil=64)
        x = resblock_conv(x, num_fil=128)
        x = resblock_base(x, num_fil=128)
        x = resblock_conv(x, num_fil=256)
        x = resblock_base(x, num_fil=256)
        x = resblock_conv(x, num_fil=512)
        x = resblock_base(x, num_fil=512)
        x = GlobalAvgPool2D()(x)
        return Model(inputs=inputs, outputs=x)(x_input)
    return bankbone


def OutputLayer(embd_shape=512, w_dacay=5e-4, name='OutputLayer'):
    """Output Layer"""
    def output_layer(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = BatchNormalization()(inputs)
        x = Dropout(rate=0.5)(x)
        x = Flatten()(x)
        x = Dense(embd_shape)(x)
        x = BatchNormalization()(x)
        return Model(inputs, x, name=name)(x_in)
    return output_layer


def NormHead(num_classes, w_decay=5e-4, name='NormHead'):
    """Norm Head"""
    def norm_head(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = Dense(num_classes, activation='softmax')(x)
        return Model(inputs, x, name=name)(x_in)
    return norm_head


def resnet(classes_num=10, embd_shape=512):
    x = inputs = Input(shape=[None, None, 3])
    x = Resnet18()(x)
    embds = OutputLayer(embd_shape=embd_shape)(x)
    outputs = NormHead(classes_num)(embds)
    return Model(inputs, outputs, name='resnet')


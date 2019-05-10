import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras.layers import (Conv3D, CuDNNGRU, MaxPool3D, Dense,
                                     Conv3DTranspose, UpSampling3D)
from tensorflow.nn import relu
from functools import reduce


class Encoder(K.Model):
    def __init__(self, features):
        super().__init__()
        self.features = features
        self.input_conv = Conv3D(4, kernel_size=(3, 3, 3),
                                 data_format="channels_first",
                                 kernel_initializer=glorot_normal(),
                                 padding="same")
        self.conv_1 = [Conv3D(feature, kernel_size=(3, 3, 3),
                                    data_format="channels_first",
                                    kernel_initializer=glorot_normal(),
                                    padding="same")
                             for feature in features]
        self.conv_2 = [Conv3D(feature, kernel_size=(3, 3, 3),
                                              data_format="channels_first",
                                              kernel_initializer=glorot_normal(),
                                              padding="same")
                       for feature in features]
        self.conv_3 = [Conv3D(feature, kernel_size=(3, 3, 3),
                                              data_format="channels_first",
                                              kernel_initializer=glorot_normal(),
                                              padding="same")
                       for feature in features]
        self.max_pool = MaxPool3D((2, 2, 2), (2, 2, 2),
                                  data_format="channels_first")

    def call(self, inputs):
        result = []
        activation_map = []

        x = self.input_conv(inputs)
        for conv_1, conv_2, conv_3 in zip(self.conv_1, self.conv_2, self.conv_3):
            x1 = conv_1(x)
            x1 = relu(x1)

            x2 = conv_2(x1)
            x2 = relu(x2)

            x3 = conv_3(x2)

            x = x3 + x1

            x = self.max_pool(x)
            result.append(x)

        return result


class LSTMFeatures(K.Model):
    def __init__(self, units, input_shape):
        super().__init__()
        self.units = units
        self.o_shape = input_shape
        self.dense = Dense(reduce((lambda x, y: x * y), input_shape),
                           kernel_initializer="glorot_uniform")
        self.gru = CuDNNGRU(reduce((lambda x, y: x * y), input_shape),
                            kernel_initializer="glorot_uniform")

    def call(self, encoder_output):
        flatten = tf.reshape(encoder_output, [16, 1, -1])
        dense = self.gru(flatten)
        output = tf.reshape(dense, (16, 1, *self.o_shape))

        return output


class Decoder(K.Model):
    def __init__(self, features):
        super().__init__()
        self.features = features
        self.input_conv = [Conv3DTranspose(feature*2, kernel_size=(3, 3, 3),
                                           data_format="channels_first",
                                           kernel_initializer=glorot_normal(),
                                           padding="same")
                           for feature in features[::-1]]
        self.up_conv = [Conv3DTranspose(feature*2, kernel_size=(3, 3, 3),
                                        data_format="channels_first",
                                        kernel_initializer=glorot_normal(),
                                        padding="same")
                        for feature in features[::-1]]
        self.final_conv = Conv3DTranspose(1, kernel_size=(3, 3, 3),
                                          data_format="channels_first",
                                          kernel_initializer=glorot_normal(),
                                          padding="same")
        self.up_pool = UpSampling3D((2, 2, 2), data_format="channels_first")

    def call(self, encoder_output, feature_space):
        x = feature_space
        for enc, up_conv, input_conv in zip(reversed(encoder_output), self.up_conv,
                                self.input_conv):
            x = tf.concat((enc, x), axis=1)
            x1 = input_conv(x)
            x1 = relu(x1)

            x2 = up_conv(x1)
            x2 = relu(x2)

            x3 = up_conv(x2)

            x = x3 + x1
            x = self.up_pool(x)

        return self.final_conv(x)


class RSUNet(K.Model):
    def __init__(self, features, units, input_shape):
        super().__init__()
        self.feature_shape = [x//(len(features)*2) for x in input_shape]
        self.encoder = Encoder(features)
        self.feature_space = LSTMFeatures(units, self.feature_shape)
        self.decoder = Decoder(features)

    def call(self, inputs):
        x = self.encoder(inputs)
        feature_space = self.feature_space(x[-1])
        x = self.decoder(x, feature_space)

        return x

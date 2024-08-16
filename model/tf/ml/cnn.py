from __future__ import absolute_import

from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers


class ConvNet(tf.keras.Model):

    def __init__(self, n_ts, n_features, n_channels=1,
                    out_dim=1, n_filters=(8, 8, 8), 
                    dropout_p=0.2):
        
        # Initialise module class
        super(ConvNet, self).__init__()

        # Define Layers
        self.conv_1 = layers.Conv2D(filters=n_filters[0],
                                    kernel_size=(2, 1),
                                    bias_initializer=tf.keras.initializers.Zeros(),
                                    activation='relu')
        
        self.conv_2 = layers.Conv2D(filters=n_filters[1],
                                    kernel_size=(1, 2),
                                    bias_initializer=tf.keras.initializers.Zeros(),
                                    activation='relu')

        self.max_pool = layers.MaxPooling2D(pool_size=(2, 2))

        self.conv_3 = layers.Conv2D(filters=n_filters[2],
                                    kernel_size=(2, 2),
                                    bias_initializer=tf.keras.initializers.Zeros(),
                                    activation='relu')
        
        self.flatten = layers.Flatten()
        self.dropout = layers.Dropout(rate=dropout_p)
        self.linear = layers.Dense(units=out_dim)
        
    
    def call(self, x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.max_pool(out)
        out = self.conv_3(out)
        out = self.dropout(self.flatten(out))
        out = self.linear(out)
        return out
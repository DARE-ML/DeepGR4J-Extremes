import torch 
import tensorflow as tf
import torch.nn as nn


class LSTM(tf.keras.Model):

    def __init__(self, window_size, input_dim, lstm_dim, hidden_dim, output_dim, n_layers, dropout=0.5):
                
        super().__init__()

        # Window Size
        # self.window_size = window_size

        # Input Dims
        self.input_dim = input_dim

        # LSTM Dims
        self.lstm_dim = lstm_dim
        self.n_layers = n_layers

        # Hidden Dims
        self.hidden_dim = hidden_dim

        # Output Dims
        self.output_dim = output_dim

        # Input layer
        # self.input_layer = tf.keras.layers.InputLayer(shape=(window_size, self.input_dim))

        # RNN layer
        self.lstm_layer = tf.keras.layers.LSTM(self.lstm_dim, 
                                                return_sequences=True,
                                                return_state=True,
                                                name='lstm_layer')

        # Flatten layer
        self.flatten = tf.keras.layers.Flatten()

        # Fully-connected output layer
        self.fc1 = tf.keras.layers.Dense(self.hidden_dim, name='dense_hidden')

        self.fc2 = tf.keras.layers.Dense(self.output_dim, name='dense_out')

        self.do = tf.keras.layers.Dropout(dropout)

        # Initialize weights
        self.init_weights()

    def call(self, x):

        # Validate input shape
        assert len(x.shape)==3, f"Expected input to be 3-dim, got {len(x.shape)}"

        # Pass through the recurrent layer
        # x = self.input_layer(x)
        out, _, _ = self.lstm_layer(x)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = self.flatten(out[:, -3:, :])
        out = tf.tanh(out)
        out = self.do(out)
        out = tf.tanh(self.fc1(out))
        out = self.fc2(out)

        return out

    
    def init_zero_hidden(self, batch_size=1) -> tf.Tensor:
        """
        Helper function.
        Returns a hidden state with specified batch size. Defaults to 1
        """
        h_0 = tf.zeros([batch_size, self.lstm_dim])
        c_0 = tf.zeros([batch_size, self.lstm_dim])
        return h_0, c_0
    

    def init_weights(self):
        for p in self.lstm_layer.trainable_variables:
            tf.keras.initializers.RandomNormal()(p)
        tf.keras.initializers.RandomNormal()(self.fc1.weights)
        tf.keras.initializers.RandomNormal()(self.fc2.weights)
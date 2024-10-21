import tensorflow as tf

class MLP(tf.keras.Model):

    def __init__(self, window_size, input_dim, hidden_dim, output_dim, n_layers, dropout=0.5):
        super().__init__()

        # Input Dims
        self.input_dim = input_dim

        # Hidden Dims
        self.hidden_dim = hidden_dim

        # Output Dims
        self.output_dim = output_dim

        # Number of layers
        self.n_layers = n_layers

        # Fully-connected hidden layers
        self.hidden_layers = [tf.keras.layers.Dense(self.hidden_dim, activation='relu', name=f'dense_hidden_{i}') for i in range(self.n_layers)]

        # Fully-connected output layer
        self.output_layer = tf.keras.layers.Dense(self.output_dim, name='dense_out')

        # Dropout layer
        self.do = tf.keras.layers.Dropout(dropout)

        # Initialize weights
        self.init_weights()

    def call(self, x):

        # Validate input shape
        assert len(x.shape) == 3, f"Expected input to be 3-dim, got {len(x.shape)}"

        # Flatten the input
        out = tf.keras.layers.Flatten()(x)

        # Pass through the hidden layers
        for layer in self.hidden_layers:
            out = layer(out)
            out = self.do(out)

        # Pass through the output layer
        out = self.output_layer(out)

        return out

    def init_weights(self):
        for layer in self.hidden_layers:
            tf.keras.initializers.RandomNormal()(layer.weights)
        tf.keras.initializers.RandomNormal()(self.output_layer.weights)
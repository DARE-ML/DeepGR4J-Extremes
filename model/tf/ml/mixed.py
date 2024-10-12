from __future__ import absolute_import

import tensorflow as tf
from .lstm import LSTM

def get_mixed_model(ts_model_name, ts_model_config, static_model_config, hidden_dim, output_dim):
    
    # Configure timeseries model
    if ts_model_name == 'lstm':
        ts_model = LSTM(window_size=ts_model_config['window_size'],
                        input_dim=ts_model_config['input_dim'],
                        hidden_dim=ts_model_config['hidden_dim'],
                        lstm_dim=ts_model_config['lstm_dim'],
                        n_layers=ts_model_config['n_layers'],
                        output_dim=ts_model_config['output_dim'],
                        dropout=ts_model_config['dropout'])
    
    elif ts_model_name == 'cnn':
        ts_model = ConvNet(n_ts=ts_model_config['window_size'],
                           n_features=ts_model_config['n_features'],
                           n_channels=ts_model_config['n_channels'],
                           out_dim=ts_model_config['output_dim'],
                           n_filters=ts_model_config['n_filters'],
                           dropout_p=ts_model_config['dropout'])

    # Configure static model
    static_model = tf.keras.Sequential([
                        tf.keras.layers.Dense(static_model_config['hidden_dim'], activation='tanh'),
                        tf.keras.layers.Dense(static_model_config['hidden_dim'], activation='tanh'),
                        tf.keras.layers.Dense(static_model_config['output_dim'], activation='relu')
                    ])

    # Define input layers
    if ts_model_name == 'lstm':
        timeseries = tf.keras.Input(shape=(ts_model_config['window_size'], ts_model_config['input_dim']), name='timeseries')
    elif ts_model_name == 'cnn':
        timeseries = tf.keras.Input(shape=(ts_model_config['window_size'], ts_model_config['input_dim'], 1), name='timeseries')
    else:
        print(ts_model_name)
    static = tf.keras.Input(shape=(static_model_config['input_dim'],), name='static')

    # Combine inputs
    relu = tf.keras.layers.Activation('relu')
    ts_hidden = relu(ts_model(timeseries))
    static_hidden = static_model(static)
    concatenated = tf.keras.layers.Concatenate()([ts_hidden, static_hidden])

    # Dense model
    hidden = tf.keras.layers.Dense(hidden_dim, activation='tanh')(concatenated)
    hidden = tf.keras.layers.Dense(hidden_dim, activation='tanh')(hidden)
    output = tf.keras.layers.Dense(output_dim, activation='linear')(hidden)

    # Combined model
    model_combined = tf.keras.Model(inputs=[timeseries, static],
                                    outputs=output)
    return model_combined

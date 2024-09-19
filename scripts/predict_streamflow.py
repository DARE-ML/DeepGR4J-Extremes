
import os
import sys
import argparse
import pickle
import datetime as dt
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm, trange
from sklearn.metrics import mean_squared_error

root_dir = '/srv/scratch/z5370003/projects/DeepGR4J-Extremes'
sys.path.append(root_dir)

from model.tf.ml import ConvNet, LSTM
from model.tf.hydro import ProductionStorage
from data.tf.camels_dataset import CamelsDataset, HybridDataset
from utils.evaluation import nse, normalize



window_size = 8

# CDF TS Model Parameters
cdf_ts_model_name = 'lstm'
cdf_ts_input_dim = 6
cdf_ts_hidden_dim = 32
cdf_ts_output_dim = 8
cdf_ts_lstm_dim = 64
cdf_ts_n_layers = 4
cdf_dropout = 0.2
# For CNN
cdf_ts_n_channels = 1
cdf_ts_n_filters = [16, 16, 8]

# CDF Static model parameters
cdf_static_input_dim = 7
cdf_static_hidden_dim = 64
cdf_static_output_dim = 8

# CDF Combined model parameters
cdf_combined_hidden_dim = 64
cdf_target_vars = ['flow_cdf']

# Quantile NN model
flow_input_dim = 6
flow_hidden_dim = 32
flow_lstm_dim = 64
flow_n_layers = 4
flow_dropout = 0.2
flow_quantiles = [0.05, 0.5, 0.95]
flow_target_vars = ['streamflow_mmd']


# Directories
camels_data_dir = '../../data/camels/aus'
gr4j_results_dir = '../results/gr4j'
cdf_results_dir = '../results/flow_cdf_saved'
deepgr4j_results_dir = '../results/qdeepgr4j_5q/aus'
gr4j_results_path = f'{gr4j_results_dir}/result.csv'



def get_ensemble_model():
    if cdf_ts_model_name == 'lstm':
        ts_model = LSTM(input_dim=cdf_ts_input_dim,
                    hidden_dim=cdf_ts_hidden_dim,
                    lstm_dim=cdf_ts_lstm_dim,
                    n_layers=cdf_ts_n_layers,
                    output_dim=cdf_ts_output_dim,
                    dropout=cdf_dropout)
    
    elif cdf_ts_model_name == 'cnn':
        ts_model = ConvNet(n_ts=window_size,
                           n_features=cdf_ts_input_dim,
                           n_channels=cdf_ts_n_channels,
                           out_dim=cdf_ts_output_dim,
                           n_filters=cdf_ts_n_filters,
                           dropout_p=cdf_dropout)

    static_model = tf.keras.Sequential([
                        tf.keras.layers.Dense(cdf_static_hidden_dim, activation='tanh'),
                        tf.keras.layers.Dense(cdf_static_hidden_dim, activation='tanh'),
                        tf.keras.layers.Dense(cdf_static_output_dim, activation='relu')
                    ])

    # Define input layers
    if cdf_ts_model_name == 'lstm':
        timeseries = tf.keras.Input(shape=(window_size, cdf_ts_input_dim), name='timeseries')
    elif cdf_ts_model_name == 'cnn':
        timeseries = tf.keras.Input(shape=(window_size, cdf_ts_input_dim, 1), name='timeseries')
    else:
        print(cdf_ts_model_name)
    static = tf.keras.Input(shape=(cdf_static_input_dim,), name='static')

    # Combine inputs
    relu = tf.keras.layers.Activation('relu')
    ts_hidden = relu(ts_model(timeseries))
    static_hidden = static_model(static)
    concatenated = tf.keras.layers.Concatenate()([ts_hidden, static_hidden])

    # Dense model
    hidden = tf.keras.layers.Dense(cdf_combined_hidden_dim, activation='tanh')(concatenated)
    hidden = tf.keras.layers.Dense(cdf_combined_hidden_dim, activation='tanh')(hidden)
    output = tf.keras.layers.Dense(len(cdf_target_vars), activation='linear')(hidden)

    # Combined model
    model_combined = tf.keras.Model(inputs=[timeseries, static],
                                    outputs=output)
    return model_combined



def generate_cdf_preds(model, dl, results_dir='../results/predictions_5q'):
    
    preds = []
    true = []
    
    for step, batch in enumerate(dl):
        out = model([batch['timeseries'], batch['static']],
                     training=False) 
        preds.append(out)
        true.append(batch['target'])
    
    preds = tf.concat(preds, axis=0)
    true = tf.concat(true, axis=0)

    # Convert to numpy array
    preds = camels_ds.target_scaler.inverse_transform(preds.numpy())
    true = camels_ds.target_scaler.inverse_transform(true.numpy())

    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(preds[:, -1], label='pred')
    ax.plot(true[:, -1], label='true')
    plt.legend()
    fig.savefig(f'{results_dir}/cdf_preds.png', bbox_inches='tight')
    
    # Calculate scores
    mse_error = mean_squared_error(true[:, -1], preds[:, -1])
    
    return mse_error, preds



def generate_flow_preds(model, dl, results_dir='../results/predictions'):
    preds = []
    true = []

    for step, batch in enumerate(dl):
        out = model(batch['timeseries'],
                    training=False) 
        preds.append(out)
        true.append(batch['target'])
    
    preds = tf.concat(preds, axis=0)
    true = tf.concat(true, axis=0)

    # Convert to numpy array
    preds = hybrid_ds.target_scaler.inverse_transform(preds.numpy())
    true = hybrid_ds.target_scaler.inverse_transform(true.numpy())

    # Clip negative values
    preds = np.clip(preds, 0, None)

    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(preds[:, int(preds.shape[-1]//2)], label='pred', c='red')
    ax.plot(true[:, -1], label='true', c='black')
    ax.fill_between(range(len(preds)), preds[:, 0], preds[:, -1], alpha=0.5, color='green')
    plt.legend()
    fig.savefig(f'{results_dir}/flow_preds.png', bbox_inches='tight')
    
    # Calculate scores
    mse_score = mean_squared_error(true[:, -1], preds[:, -2])
    nse_score = nse(true[:, -1], preds[:, -2])
    nnse_score = normalize(nse_score)
    
    return mse_score, nse_score, nnse_score, preds, true


def select_by_quantiles(values_array, quantiles):
    indices = np.ones_like(quantiles)
    if values_array.shape[-1] == 3:
        indices[quantiles > 0.95] = 2
        indices[(quantiles <= 0.95) & (quantiles > 0.05)] = 1
        indices[quantiles <= 0.05] = 0
        indices = indices.astype(int).flatten()
    elif values_array.shape[-1] == 5:
        indices[(quantiles > 0.75)] = 4
        indices[(quantiles <= 0.75) & (quantiles > 0.50)] = 3
        indices[(quantiles <= 0.50) & (quantiles > 0.25)] = 2
        indices[(quantiles <= 0.25) & (quantiles > 0.05)] = 1
        indices[(quantiles <= 0.05)] = 0
    indices = indices.astype(int).flatten()
    selected_values = values_array[np.arange(len(indices)), indices]
    return selected_values





if __name__ == '__main__':

    # CDF Dataset
    camels_ds = CamelsDataset(data_dir=camels_data_dir, 
                            target_vars=cdf_target_vars,
                            window_size=window_size)
    
    zone_list = camels_ds.get_zones()

    results_all = []

    for (state_outlet, map_zone) in zone_list:
        print(f'Running CDF model for {state_outlet} - {map_zone}')

        # CDF model and scaler path
        cdf_model_path = f'{cdf_results_dir}/{state_outlet}_{map_zone}/model.keras'
        cdf_data_scaler_path = f'{cdf_results_dir}/{state_outlet}_{map_zone}/scalers.pkl'
        camels_ds.load_scalers(cdf_data_scaler_path)

        # Load CDF model for the zone
        cdf_model = get_ensemble_model()
        cdf_model.load_weights(cdf_model_path)
        cdf_model.summary()

        # Filter stations for the zone
        station_list = camels_ds.get_station_list(state_outlet=state_outlet, map_zone=map_zone)

        for station_id in station_list:
    
            print(f'Running predictions for {station_id}')
            flow_data_scaler_path = f'{deepgr4j_results_dir}/model/{station_id}/scalers.pkl'
            flow_model_path = f'{deepgr4j_results_dir}/model/{station_id}/model.keras'
            results_dir = f'../results/predictions/{station_id}'
            os.makedirs(results_dir, exist_ok=True)

            # Prepare flow CDF data
            print('Preparing CDF data')
            camels_ds.prepare_data(station_list=[station_id])
            cdf_train_ds, cdf_test_ds = camels_ds.get_datasets()

            # Generate CDF predictions
            print('Generating CDF predictions')
            _, cdf_preds = generate_cdf_preds(cdf_model, cdf_test_ds.batch(256), results_dir=results_dir)


            # Quantile NN model
            print('Preparing flow data')
            prod = ProductionStorage()

            hybrid_ds = HybridDataset(data_dir=camels_data_dir,
                                    gr4j_logfile=gr4j_results_path,
                                    prod=prod,
                                    window_size=window_size,
                                    target_vars=flow_target_vars)

            flow_model = LSTM(input_dim=flow_input_dim,
                            hidden_dim=flow_hidden_dim,
                            lstm_dim=flow_lstm_dim,
                            n_layers=flow_n_layers,
                            output_dim=len(flow_quantiles),
                            dropout=flow_dropout)

            try:
                hybrid_ds.load_scalers(flow_data_scaler_path)
                hybrid_ds.prepare_data(station_list=[station_id])
            except:
                continue
            hybrid_train_ds, hybrid_test_ds = hybrid_ds.get_datasets()

            # Load flow model weights
            flow_model.summary()
            print(flow_model_path)
            flow_model.load_weights(flow_model_path)
            generate_flow_preds(flow_model, hybrid_test_ds.batch(256))
            flow_model.load_weights(flow_model_path)
            flow_model.summary()

            # Generate flow predictions
            print('Generating flow predictions')
            res = generate_flow_preds(flow_model, hybrid_test_ds.batch(256), results_dir=results_dir)
            mse_score, nse_score, nnse_score, qpreds, true = res
            print(f'MSE: {mse_score}, NSE: {nse_score}, NNSE: {nnse_score}')


            # Corrected preds using flow CDF
            flowpreds = select_by_quantiles(qpreds, cdf_preds)
            mse_score_corrected = mean_squared_error(true.flatten(), flowpreds)
            nse_score_corrected = nse(true.flatten(), flowpreds)
            nnse_score_corrected = normalize(nse(true.flatten(), flowpreds))
            nse_score_corrected, nnse_score_corrected

            print(f'Corrected NSE: {nse_score_corrected}, Corrected NNSE: {nnse_score_corrected}')

            results_all.append({
                'station_id': station_id,
                'mse_score': mse_score,
                'nse_score': nse_score,
                'nnse_score': nnse_score,
                'mse_score_corrected': mse_score_corrected,
                'nse_score_corrected': nse_score_corrected,
                'nnse_score_corrected': nnse_score_corrected
            })

    results_df = pd.DataFrame(results_all)
    results_df.to_csv('../results/predictions/results.csv', index=False)


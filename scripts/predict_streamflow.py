
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

from model.tf.ml import ConvNet, LSTM, get_mixed_model
from model.tf.hydro import ProductionStorage
from data.tf.camels_dataset import CamelsDataset, HybridDataset
from utils.evaluation import nse, normalize


# Window size
cdf_window_size = 8
flow_window_size = 8

# CDF TS Model Parameters
cdf_ts_model_name = 'lstm'

cdf_lstm_config = {
    'n_steps': cdf_window_size,
    'input_dim': 6,
    'hidden_dim': 32,
    'output_dim': 8,
    'lstm_dim': 64,
    'n_layers': 4,
    'dropout': 0.2
}

cdf_cnn_config = {
    'n_steps': cdf_window_size,
    'n_features': 6,
    'n_channels': 1,
    'output_dim': 8,
    'n_filters': [16, 16, 8],
    'dropout': 0.2
}

# CDF Static model parameters
cdf_static_config = {
    'input_dim': 7,
    'hidden_dim': 64,
    'output_dim': 8
}

# CDF Combined model parameters
cdf_combined_hidden_dim = 64
cdf_target_vars = ['flow_cdf']
cdf_output_dim = len(cdf_target_vars)

# Quantile NN model
flow_input_dim = 10
flow_hidden_dim = 32
flow_lstm_dim = 64
flow_n_layers = 4
flow_dropout = 0.2
flow_quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
flow_target_vars = ['streamflow_mmd']



# Directories
camels_data_dir = '../../data/camels/aus'
gr4j_results_dir = '../results/gr4j'
cdf_results_dir = '../results/flow_cdf_saved_w8'
deepgr4j_results_dir = '../results/qdeepgr4j_5q_w8/aus'
gr4j_results_path = f'{gr4j_results_dir}/result.csv'
results_dir = '../results/prediction_5q_w8'




def generate_cdf_preds(model, dl, results_dir='../results/predictions'):
    
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
    median_index = int(preds.shape[-1]//2)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(preds[:, median_index], label='pred', c='red')
    ax.plot(true[:, -1], label='true', c='black')
    ax.fill_between(range(len(preds)), preds[:, 0], preds[:, -1], alpha=0.5, color='green')
    plt.legend()
    fig.savefig(f'{results_dir}/flow_preds.png', bbox_inches='tight')
    
    # Calculate scores
    mse_score = mean_squared_error(true[:, -1], preds[:, median_index])
    nse_score = nse(true[:, -1], preds[:, median_index])
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
                            window_size=cdf_window_size)
    
    zone_list = camels_ds.get_zones()

    results_all = []

    for (state_outlet, map_zone) in zone_list:
        print(f'Running CDF model for {state_outlet} - {map_zone}')

        # CDF model and scaler path
        cdf_model_path = f'{cdf_results_dir}/{state_outlet}_{map_zone}/model.keras'
        cdf_data_scaler_path = f'{cdf_results_dir}/{state_outlet}_{map_zone}/scalers.pkl'
        camels_ds.load_scalers(cdf_data_scaler_path)

        # Load CDF model for the zone
        cdf_model = get_mixed_model(cdf_ts_model_name, cdf_lstm_config, cdf_static_config, cdf_combined_hidden_dim, cdf_output_dim)
        cdf_model.load_weights(cdf_model_path)
        cdf_model.summary()

        # Filter stations for the zone
        station_list = camels_ds.get_station_list(state_outlet=state_outlet, map_zone=map_zone)

        for station_id in station_list:

            # if station_id != '102101A':
            #     continue
    
            print(f'Running predictions for {station_id}')
            flow_data_scaler_path = f'{deepgr4j_results_dir}/model/{station_id}/scalers.pkl'
            flow_model_path = f'{deepgr4j_results_dir}/model/{station_id}/flow_model.weights.h5'
            station_results_dir = f'{results_dir}/{station_id}'
            os.makedirs(station_results_dir, exist_ok=True)

            # Prepare flow CDF data
            print('Preparing CDF data')
            camels_ds.prepare_data(station_list=[station_id])
            cdf_train_ds, cdf_test_ds = camels_ds.get_datasets()
            print('Generating CDF predictions')
            _, cdf_preds = generate_cdf_preds(cdf_model, cdf_test_ds.batch(256), results_dir=station_results_dir)


            # Quantile NN model
            print('Preparing flow data')
            prod = ProductionStorage()

            hybrid_ds = HybridDataset(data_dir=camels_data_dir,
                                      gr4j_logfile=gr4j_results_path,
                                      prod=prod,
                                      window_size=flow_window_size,
                                      target_vars=flow_target_vars)

            flow_model = LSTM(window_size=flow_window_size,
                              input_dim=flow_input_dim,
                              hidden_dim=flow_hidden_dim,
                              lstm_dim=flow_lstm_dim,
                              n_layers=flow_n_layers,
                              output_dim=len(flow_quantiles),
                              dropout=flow_dropout)

            # Load flow model weights
            flow_model(tf.random.uniform(shape=(5, flow_window_size, flow_input_dim)))
            flow_model.summary()
            flow_model.load_weights(flow_model_path)

            try:
                hybrid_ds.load_scalers(flow_data_scaler_path)
                hybrid_ds.prepare_data(station_list=[station_id])
            except:
                continue
            hybrid_train_ds, hybrid_test_ds = hybrid_ds.get_datasets()


            # Generate flow predictions
            print('Generating flow predictions')
            res = generate_flow_preds(flow_model, hybrid_test_ds.batch(256), results_dir=station_results_dir)
            mse_score, nse_score, nnse_score, qpreds, true = res
            print(f'\nMSE: {mse_score:.4f}, NSE: {nse_score:.4f}, NNSE: {nnse_score:.4f}')


            # Corrected preds using flow CDF
            flowpreds = select_by_quantiles(qpreds, cdf_preds)
            mse_score_corrected = mean_squared_error(true.flatten(), flowpreds)
            nse_score_corrected = nse(true.flatten(), flowpreds)
            nnse_score_corrected = normalize(nse(true.flatten(), flowpreds))
            nse_score_corrected, nnse_score_corrected

            print(f'\nCorrected MSE: {mse_score_corrected:.4f} Corrected NSE: {nse_score_corrected:.4f}, Corrected NNSE: {nnse_score_corrected:.4f}')

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
    results_df.to_csv(os.path.join(results_dir, 'results.csv'), index=False)


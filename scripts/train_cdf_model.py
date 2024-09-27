import os
import sys
import time
import argparse
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
from data.tf.camels_dataset import CamelsDataset
from utils.training import EarlyStopper, Trainer

# Create argument parser
parser = argparse.ArgumentParser(description='Train Flow CDF Model')

# Data processing parameters
parser.add_argument('--camels_dir', type=str, default='../../data/camels/aus', help='Directory containing CAMELS dataset')
parser.add_argument('--target_vars', type=str, nargs='+', default=['flow_cdf'], help='Target variables')
parser.add_argument('--state_outlet', type=str, default=None, help='State outlet')
parser.add_argument('--map_zone', type=int, default=None, help='Map zone')
parser.add_argument('--station_id', type=str, nargs='+', default=None, help='Target variables')
parser.add_argument('--window_size', type=int, default=10, help='Size of the sliding window')
parser.add_argument('--results_dir', type=str, default='../results/flow_cdf', help='Directory to save results')
parser.add_argument('--ts_model', type=str, default='lstm', help='Time series model')

# Timeseries model parameters
parser.add_argument('--ts_input_dim', type=int, default=6, help='Input dimension for timeseries model')
parser.add_argument('--ts_output_dim', type=int, default=8, help='Output dimension for timeseries model')
parser.add_argument('--ts_dropout', type=float, default=0.2, help='Dropout rate for timeseries model')

# LSTM model parameters
parser.add_argument('--ts_hidden_dim', type=int, default=32, help='Hidden dimension for LSTM timeseries model')
parser.add_argument('--ts_lstm_dim', type=int, default=64, help='LSTM dimensions')
parser.add_argument('--ts_n_layers', type=int, default=4, help='Number of LSTM layers')

# CNN model parameters
parser.add_argument('--ts_n_channels', type=int, default=1, help='Number of input channels for CNN timeseries model')
parser.add_argument('--ts_n_filters', type=int, nargs='+', default=[16, 16, 8], help='Number of filters for CNN timeseries model')

# Static model parameters
parser.add_argument('--static_input_dim', type=int, default=7, help='Input dimension for static model')
parser.add_argument('--static_hidden_dim', type=int, default=64, help='Hidden dimension for static model')
parser.add_argument('--static_output_dim', type=int, default=8, help='Output dimension for static model')

# Combined model parameters
parser.add_argument('--combined_hidden_dim', type=int, default=64, help='Hidden dimension for combined model')

# Training parameters
parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
parser.add_argument('--verbose', type=int, default=1, help='Verbosity level')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
parser.add_argument('--beta_1', type=float, default=0.88, help='Beta 1 for Adam optimizer')
parser.add_argument('--beta_2', type=float, default=0.997, help='Beta 2 for Adam optimizer')
parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay for Adam optimizer')
parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
parser.add_argument('--min_delta', type=float, default=0.01, help='Minimum change in validation loss for early stopping')



def plot_timeseries(model, dl, loss_fn, results_dir):
    preds = []
    true = []
    stations = []
    for step, batch in enumerate(dl):
        # Run the forward pass of the layer.
        if args.ts_model == 'cnn':
            batch['timeseries'] = tf.expand_dims(batch['timeseries'], axis=-1)
        out = model([batch['timeseries'], batch['static']],
                     training=False) 
        preds.append(out)
        true.append(batch['target'])
        stations.append(batch['station_id'])
    preds = tf.concat(preds, axis=0)
    true = tf.concat(true, axis=0)
    stations = tf.concat(stations, axis=0)

    print(f"{results_dir} loss: {loss_fn(true, preds).numpy():.4f}")

    # Convert to numpy array
    preds = camels_ds.target_scaler.inverse_transform(preds.numpy())
    true = camels_ds.target_scaler.inverse_transform(true.numpy())
    stations = stations.numpy().flatten()

    os.makedirs(results_dir, exist_ok=True)
    mse_error = {}

    for station in np.unique(stations):
        idx = (stations==station)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(preds[idx, -1], label='pred')
        ax.plot(true[idx, -1], label='true')
        mse_error[station] = mean_squared_error(true[idx, -1], preds[idx, -1])
        plt.legend()
        plt.savefig(os.path.join(results_dir, f'{station.decode("utf-8")}.png'))
        plt.close()
    
    return mse_error


def get_ts_model_config(args):
    if args.ts_model == 'lstm':
        ts_model_config = {
            'window_size': args.window_size,
            'input_dim': args.ts_input_dim,
            'output_dim': args.ts_output_dim,
            'hidden_dim': args.ts_hidden_dim,
            'lstm_dim': args.ts_lstm_dim,
            'n_layers': args.ts_n_layers,
            'dropout': args.ts_dropout
        }
    elif args.ts_model == 'cnn':
        ts_model_config = {
            'n_ts': args.winfow_size,
            'input_dim': args.ts_input_dim,
            'output_dim': args.ts_output_dim,
            'n_channels': args.ts_n_channels,
            'n_filters': args.ts_n_filters,
            'dropout': args.ts_dropout
        }
    else:
        raise ValueError(f"Invalid timeseries model: {args.ts_model}")
    
    return ts_model_config





if __name__ == '__main__':
    
    # Parse arguments
    args = parser.parse_args()
    print(args)

    # Get timeseries model config
    ts_model_config = get_ts_model_config(args)

    # Define static model config
    static_model_config = {
        'input_dim': args.static_input_dim,
        'hidden_dim': args.static_hidden_dim,
        'output_dim': args.static_output_dim
    }
    output_dim = len(args.target_vars)

    # Load dataset
    camels_ds = CamelsDataset(data_dir=args.camels_dir, 
                              target_vars=args.target_vars,
                              window_size=args.window_size)
    zone_list = camels_ds.get_zones()

    for state_outlet, map_zone in zone_list:

        # Assign state_outlet and map_zone
        args.state_outlet = state_outlet
        args.map_zone = map_zone

        # Prepare data
        camels_ds.prepare_data(station_list=args.station_id,
                                state_outlet=args.state_outlet,
                                map_zone=args.map_zone)
        
        # Get datasets
        train_ds, test_ds = camels_ds.get_datasets()
        print(camels_ds.station_list)


        model = get_mixed_model(args.ts_model, ts_model_config,
                                static_model_config, combined_hidden_dim,
                                output_dim)
        model.summary()

        # Define optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate,
                                            beta_1=args.beta_1,
                                            beta_2=args.beta_2,
                                            weight_decay=args.weight_decay,
                                            epsilon=1e-07,
                                            amsgrad=False)

        # Define loss function
        loss_fn = tf.keras.losses.MeanSquaredError()

        # Define early stopper
        early_stopper = EarlyStopper(patience=args.patience, min_delta=args.min_delta)

        # Define trainer
        trainer = Trainer(model, optimizer, loss_fn=loss_fn, 
                          early_stopper=early_stopper, 
                          model_type='ensemble')

        # Train model
        model, train_losses, test_losses = trainer.train(train_ds, test_ds, args)

        # Plot timeseries predictions
        if (args.state_outlet is not None) or (args.map_zone is not None):
            results_dir = os.path.join(args.results_dir, f'{args.state_outlet}_{args.map_zone}')
        else:
            results_dir = os.path.join(args.results_dir, 'aus')

        os.makedirs(args.results_dir, exist_ok=True)
        train_mse_error = plot_timeseries(model, train_ds.batch(args.batch_size),
                                          loss_fn, os.path.join(results_dir, 'training'))

        test_mse_error = plot_timeseries(model, test_ds.batch(args.batch_size),
                                         loss_fn, os.path.join(results_dir, 'testing'))

        # Save results
        results = pd.DataFrame({'train_mse': train_mse_error, 'test_mse': test_mse_error})
        results.index.name = 'station_id'
        results = results.reset_index()
        results.to_csv(os.path.join(results_dir, 'results.csv'))

        # Create array for losses
        train_losses = np.array(train_losses)
        test_losses = np.array(test_losses)

        # Plot losses against epochs
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(train_losses, label='train')
        ax.plot(test_losses, label='test')
        plt.legend()
        plt.savefig(os.path.join(results_dir, 'losses.png'))

        # Save model
        tf.keras.models.save_model(model, os.path.join(results_dir, 'model.keras'), overwrite=True)
        print(f"Model saved to {results_dir}")

        # Save scalers
        camels_ds.save_scalers(results_dir)
        print(f"Scalers saved to {results_dir}")


        

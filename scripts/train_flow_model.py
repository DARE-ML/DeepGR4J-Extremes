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
from sklearn.metrics import mean_squared_error, r2_score

root_dir = '/srv/scratch/z5370003/projects/DeepGR4J-Extremes'
sys.path.append(root_dir)

from model.tf.ml import ConvNet, LSTM, RNN, MLP
from model.tf.hydro import ProductionStorage
from data.tf.camels_dataset import CamelsDataset, HybridDataset
from utils.training import EarlyStopper, Trainer, TiltedLossMultiQuantile
from utils.evaluation import nse, normalize, confidence_score

# Create argument parser
parser = argparse.ArgumentParser(description='Train Quantile Flow Model')

# Add arguments
parser.add_argument('--window_size', type=int, default=10, help='Size of the sliding window')
parser.add_argument('--camels_dir', type=str, default='../../data/camels/aus', help='Directory containing CAMELS dataset')
parser.add_argument('--gr4j_logfile', type=str, default='../results/gr4j/result.csv', help='GR4J calibration results file')
parser.add_argument('--station_id', type=str, nargs='+', default=None, help='Station ID')
parser.add_argument('--results_dir', type=str, default='../results/flow_cdf', help='Directory to save results')
parser.add_argument('--state_outlet', type=str, default=None, help='State outlet')
parser.add_argument('--map_zone', type=int, default=None, help='Map zone')
parser.add_argument('--ts_model', type=str, default='lstm', help='Time series model')

parser.add_argument('--hidden_dim', type=int, default=32, help='Hidden dimension for LSTM')
parser.add_argument('--lstm_dim', type=int, default=64, help='LSTM dimension')
parser.add_argument('--rnn_dim', type=int, default=64, help='RNN dimension')
parser.add_argument('--n_layers', type=int, default=4, help='Number of LSTM layers')
parser.add_argument('--ts_output_dim', type=int, default=8, help='Output dimension for LSTM')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate for LSTM')

parser.add_argument('--n_channels', type=int, default=1, help='Number of channels')
parser.add_argument('--n_filters', type=int, nargs='+', default=[16, 16, 8], help='Number of filters for CNN')
parser.add_argument('--cnn_dropout', type=float, default=0.1, help='Dropout rate for CNN')

parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
parser.add_argument('--verbose', type=int, default=1, help='Verbosity level')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--beta_1', type=float, default=0.89, help='Beta 1 for Adam optimizer')
parser.add_argument('--beta_2', type=float, default=0.97, help='Beta 2 for Adam optimizer')
parser.add_argument('--weight_decay', type=float, default=2e-2, help='Weight decay for Adam optimizer')
parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
parser.add_argument('--min_delta', type=float, default=0.01, help='Minimum change in validation loss for early stopping')


def get_model(args):
    if args.ts_model == 'lstm':
        ts_model = LSTM(window_size=args.window_size,
                        input_dim=args.ts_input_dim,
                        hidden_dim=args.hidden_dim,
                        lstm_dim=args.lstm_dim,
                        n_layers=args.n_layers,
                        output_dim=len(args.quantiles),
                        dropout=args.dropout)
    
    elif args.ts_model == 'cnn':
        ts_model = ConvNet(n_ts=args.window_size,
                           n_features=args.ts_input_dim,
                           n_channels=args.n_channels,
                           out_dim=len(args.quantiles),
                           n_filters=args.n_filters,
                           dropout_p=args.dropout)
        
    elif args.ts_model == 'rnn':
        ts_model = RNN(window_size=args.window_size,
                       input_dim=args.ts_input_dim,
                       hidden_dim=args.hidden_dim,
                       rnn_dim=args.rnn_dim,
                       output_dim=len(args.quantiles),
                       n_layers=args.n_layers,
                       dropout=args.dropout)
    
    elif args.ts_model == 'mlp':
        ts_model = MLP(window_size=args.window_size,
                       input_dim=args.ts_input_dim,
                       hidden_dim=args.hidden_dim,
                       output_dim=len(args.quantiles),
                       n_layers=args.n_layers,
                       dropout=args.dropout)
    
    else:
        raise ValueError(f"Invalid time series model: {args.ts_model}")

    return ts_model

def generate_predictions(model, dl, loss_fn, results_dir):
    preds = []
    true = []
    stations = []
    for step, batch in enumerate(dl):
        # Run the forward pass of the layer.
        if args.ts_model == 'cnn':
            batch['timeseries'] = tf.expand_dims(batch['timeseries'], axis=-1)
        out = model(batch['timeseries'],
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

    # Clip negative values
    preds = np.clip(preds, 0, None)
    stations = stations.numpy().flatten()

    os.makedirs(results_dir, exist_ok=True)
    mse_score = {}
    nse_score = {}
    nnse_score = {}
    conf_score = {}
    rsquared_score = {}
    
    for station in np.unique(stations):
        idx = (stations==station)
        fig, ax = plt.subplots(figsize=(16, 6))
        n_outputs = preds.shape[-1]
        ax.plot(true[idx, -1], alpha=0.75, label=f"True", color='black')
        if n_outputs > 1:
            # for i in range(n_outputs):
            print(int(n_outputs//2))
            ax.plot(preds[idx, int(n_outputs//2)], alpha=0.65, label=f"Pred-mediam", color='red')
            ax.fill_between(range(len(preds)), preds[:, 0], preds[:, -1], alpha=0.5, color='green', label='90% conf')
        else:
            ax.plot(preds[idx, 0], alpha=0.65, label=f"Pred")
        ax.set_title(f'{station}')
        ax.set_ylabel('Streamflow (mm/day)')
        ax.set_xlabel('Time')
        plt.legend()
        plt.savefig(os.path.join(results_dir, f'{station.decode("utf-8")}.png'), bbox_inches='tight')
        plt.close()
        
        # Compute Scores
        mse_score[station] = []
        nse_score[station] = []
        nnse_score[station] = []
        rsquared_score[station] = []
        
        for j in range(n_outputs):
            mse_score[station].append(mean_squared_error(true[idx, -1], preds[idx, j]))
            rsquared_score[station].append(r2_score(true[idx, -1], preds[idx, j]))
            nse_score[station].append(nse(true[idx, -1], preds[idx, j]))
            nnse_score[station].append(normalize(nse(true[idx, -1], preds[idx, j])))
        conf_score[station] = confidence_score(true[idx, -1], preds[idx])
    
    return mse_score, rsquared_score, nse_score, nnse_score, conf_score




if __name__ == '__main__':
    
    # Parse arguments
    args = parser.parse_args()
    ts_vars = ['precipitation_AWAP', 'et_morton_actual_SILO',
           'tmax_awap', 'tmin_awap', 'vprp_awap']
    ts_hybrid_vars_len = 4
    args.ts_input_dim = len(ts_vars) + ts_hybrid_vars_len
    print(args)

    # Quantiles
    args.quantiles = tf.convert_to_tensor([0.05, 0.5, 0.95])

    # Load dataset
    camels_ds = CamelsDataset(data_dir=args.camels_dir,
                              window_size=args.window_size)
    station_list = camels_ds.get_station_list()
    # station_list = ['112102A', '121001A', '238208', '419005', 'G0050115',
    #                 '102101A', '401217', '401009', '314213', 'G9030124']

    results_all = []

    for station_id in station_list:

        # Production Storage
        prod = ProductionStorage()

        # Assign state_outlet and map_zone
        args.station_id = [station_id]

        camels_ds = HybridDataset(data_dir=args.camels_dir,
                                  gr4j_logfile=args.gr4j_logfile,
                                  prod=prod,
                                  ts_vars=ts_vars,
                                  target_vars=['streamflow_mmd'],
                                  window_size=args.window_size)

        # Prepare data
        camels_ds.prepare_data(station_list=args.station_id,
                               state_outlet=args.state_outlet,
                               map_zone=args.map_zone)
        
        
        # Get datasets
        train_ds, test_ds = camels_ds.get_datasets(test_size=0.3)
        print(f"Station Id: {args.station_id}")
        print(f"X1 after data prep: {camels_ds.prod.get_x1()}\n")

        # Get model
        model = get_model(args)
        if args.ts_model == 'cnn':
            model(tf.random.normal((args.batch_size, args.window_size, args.ts_input_dim, 1), dtype=tf.float32))
        else:
            model(tf.random.normal((args.batch_size, args.window_size, args.ts_input_dim), dtype=tf.float32))
            
        model.summary()

        # Define optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate,
                                            beta_1=args.beta_1,
                                            beta_2=args.beta_2,
                                            weight_decay=args.weight_decay,
                                            epsilon=1e-08,
                                            amsgrad=False)

        # Define loss function
        loss_fn = TiltedLossMultiQuantile(quantiles=args.quantiles)

        # Define early stopper
        early_stopper = EarlyStopper(patience=args.patience, min_delta=args.min_delta)

        # Define trainer
        trainer = Trainer(model, optimizer=optimizer, loss_fn=loss_fn,
                          model_type='ts', early_stopper=early_stopper)

        # Train model
        model, train_losses, test_losses = trainer.train(train_ds, test_ds, args)


        # Plot timeseries predictions
        if (args.state_outlet is not None) or (args.map_zone is not None):
            results_dir = os.path.join(args.results_dir, f'{args.state_outlet}_{args.map_zone}')
        else:
            results_dir = os.path.join(args.results_dir, 'aus')

        os.makedirs(args.results_dir, exist_ok=True)
        
        train_mse, train_r2, train_nse, train_nnse, train_conf = generate_predictions(
            model, 
            train_ds.batch(args.batch_size),
            loss_fn, os.path.join(results_dir, 'training')
        )

        test_mse, test_r2, test_nse, test_nnse, test_conf = generate_predictions(
            model, 
            test_ds.batch(args.batch_size),
            loss_fn, os.path.join(results_dir, 'testing')
        )

        print(test_nse, test_nnse)

        # Save results
        results = pd.DataFrame({
            'train_mse': train_mse, 
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_nse': train_nse,
            'test_nse': test_nse,
            'train_nnse': train_nnse,
            'test_nnse': test_nnse,
            'train_conf': train_conf,
            'test_conf': test_conf
        })
        results.index.name = 'station_id'
        results = results.reset_index()
        results['station_id'] = results['station_id'].apply(lambda x: x.decode('utf-8'))

        results_all.append(results)

        # Save model
        model_path = os.path.join(results_dir, 'model', station_id.encode().decode('utf-8'))
        os.makedirs(model_path, exist_ok=True)
        # model.save(os.path.join(model_path, 'model.keras'))
        model.summary()
        model.save_weights(os.path.join(model_path, 'flow_model.weights.h5'), overwrite=True)
        prod.save_weights(os.path.join(model_path, 'prod.weights.h5'), overwrite=True)

        # Save scalers
        camels_ds.save_scalers(os.path.join(model_path))

        
        # Create array for losses
        train_losses = np.array(train_losses)
        test_losses = np.array(test_losses)
    
        # Plot losses against epochs
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(train_losses, label='train')
        ax.plot(test_losses, label='test')
        plt.legend()
        plt.savefig(os.path.join(model_path, 'losses.png'))
        
    results = pd.concat(results_all)
    results.to_csv(os.path.join(results_dir, 'results.csv'))



        

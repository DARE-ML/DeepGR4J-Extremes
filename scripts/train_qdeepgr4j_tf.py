
import argparse
import datetime as dt
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
import tensorflow as tf
import tensorflow.data as tfdata

sys.path.append("../")

from model.tf.hydro.gr4j_prod import ProductionStorage
from model.tf.ml import ConvNet, LSTM
from utils.training import EarlyStopper
from utils.evaluation import evaluate
from data.tf.utils import read_dataset_from_file, get_station_list





# Create parser
parser = argparse.ArgumentParser(description="Train Quantile DeepGR4J model")

parser.add_argument('--data-dir', type=str, default='/data/camels/aus/')
parser.add_argument('--sub-dir', type=str, required=True)
parser.add_argument('--station-id', type=str, default=None)
parser.add_argument('--run-dir', type=str, default='/project/results/qdeepgr4j')
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--n-epoch', type=int, default=300)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--weight-decay', type=float, default=0.02)
parser.add_argument('--model', type=str, default='cnn')
parser.add_argument('--n-filters', nargs='+', type=int)
parser.add_argument('--input-dim', type=int, default=9)
parser.add_argument('--hidden-dim', type=int, default=32)
parser.add_argument('--lstm-dim', type=int, default=64)
parser.add_argument('--n-layers', type=int, default=2)
parser.add_argument('--quantiles', nargs='+', type=float)
parser.add_argument('--n-features', type=int, default=9)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--gr4j-run-dir', type=str, default='/project/results/gr4j')
parser.add_argument('--window-size', type=int, default=7)
parser.add_argument('--q-in', action='store_true')



def tilted_loss(y, f):
    for i, q in enumerate(args.quantiles):
        e = y[:, 0] - f[:, i]
        if i == 0:
            l = tf.reduce_mean(tf.maximum(q * e, (q - 1) * e))
        else:
            l += tf.reduce_mean(tf.maximum(q * e, (q - 1) * e))
    return l


def create_sequence(X, y, window_size, q_in):

        assert window_size is not None, "Window size cannot be NoneType."

        # Create empyty sequences
        Xs, ys = [], []

        if q_in:
            # Add sequences to Xs and ys
            for i in range(1, len(X) - window_size):
                Xs.append(tf.concat([
                                        X[i: (i + window_size)], 
                                        y[i-1: (i + window_size - 1)]
                                    ], axis=1)
                        )
                ys.append(y[i + window_size - 1])
        else:
            # Add sequences to Xs and ys
            for i in range(len(X)-window_size):
                Xs.append(X[i: (i + window_size)])
                ys.append(y[i + window_size-1])

        Xs, ys = tf.stack(Xs), tf.stack(ys)
        if 'cnn' in args.model:
            Xs = tf.expand_dims(Xs, axis=-1)

        return Xs, ys


def out_of_interval(targets, predictions, low_idx=0, high_idx=None):
    assert (len(predictions.shape)==2), "Expect 2 dimensions in predictions"
    if high_idx is None:
        high_idx = predictions.shape[-1] - 1
    out_of_interval = (targets < predictions[:, low_idx]) | (targets > predictions[:, high_idx])
    return out_of_interval.sum()/out_of_interval.shape[0]


def evaluate_preds(model, prod_store, ds, batch_size, y_mu, y_sigma, q_in, quantiles=[0.5]):
    # Empty list to store batch-wise tensors
    P = []
    ET = []
    Q = []
    Q_hat = []

    # Create a batched dataset
    ds_batched = ds.batch(batch_size)

    for X, y in ds_batched:
        # Predict using the model
        y_hat = model(X) * y_sigma + y_mu
        y = y * y_sigma + y_mu

        # Append predictions and true values to the lists
        Q.append(y.numpy())
        Q_hat.append(y_hat.numpy())

        # Invert the predictions
        if 'conv' in model.__class__.__name__.lower():
            if q_in:
                X_inv = X[:, -1, :-1, 0]*prod_store.sigma+prod_store.mu
            else:
                X_inv = X[:, -1, :, 0]*prod_store.sigma+prod_store.mu
        elif 'lstm' in model.__class__.__name__.lower():
            if q_in:
                X_inv = X[:, -1, :-1]*prod_store.sigma+prod_store.mu
            else:
                X_inv = X[:, -1]*prod_store.sigma+prod_store.mu
        
        P.append(X_inv[:, 0])
        ET.append(X_inv[:, 1])
    
    # Concatenate all batches
    P = np.concatenate(P, axis=0)
    ET = np.concatenate(ET, axis=0)
    Q = np.concatenate(Q, axis=0).flatten()
    Q_hat = np.concatenate(Q_hat, axis=0)
    Q_hat = np.clip(Q_hat, 0, None)

    return (evaluate(P, ET, Q, Q_hat, quantiles=quantiles),
            out_of_interval(targets=Q, predictions=Q_hat))



def train_and_evaluate(train_ds, val_ds,
                        station_id, n_epoch=100, 
                        batch_size=256, lr=0.001,
                        run_dir='/project/results/qdeepgr4j_cnn',
                        gr4j_run_dir='/project/results/gr4j',
                        **kwargs):

    # Get tensors from dataset
    t_train, X_train, y_train = train_ds._tensors
    t_val, X_val, y_val = val_ds._tensors

    # Handle nan values
    X_train = tf.keras.ops.nan_to_num(X_train)
    X_val = tf.keras.ops.nan_to_num(X_val)

    # Mean and std 
    y_mu = y_train.numpy().mean(axis=0)
    y_sigma = y_train.numpy().std(axis=0)

    # Scale labels
    y_train = (y_train - y_mu)/y_sigma
    y_val = (y_val - y_mu)/y_sigma

    # Read GR4J results
    gr4j_results_df = pd.read_csv(os.path.join(gr4j_run_dir, 'result.csv')).reset_index()
    gr4j_results_df.station_id = gr4j_results_df.station_id.astype(str)
    x1 = gr4j_results_df.loc[gr4j_results_df.station_id==station_id, 'x1'].values[0]

    # Create production storage instance
    prod_store = ProductionStorage(x1=x1)
    inp_train = prod_store(X_train, include_x=True)[0]
    inp_val = prod_store(X_val, include_x=True)[0]

    # Create Input sequence
    X_train, y_train = create_sequence(inp_train, y_train, 
                                       window_size=kwargs['window_size'],
                                       q_in=kwargs['q_in'])
    X_val, y_val = create_sequence(inp_val, y_val, 
                                   window_size=kwargs['window_size'], 
                                   q_in=kwargs['q_in'])

    # Create Sequence Datasets and DataLoaders 
    train_ds = tfdata.Dataset.from_tensor_slices((X_train, y_train))
    train_dl = train_ds.shuffle(buffer_size=1024).batch(batch_size=batch_size)

    val_ds = tfdata.Dataset.from_tensor_slices((X_val, y_val))
    val_dl = val_ds.shuffle(buffer_size=1024).batch(batch_size=batch_size)


    # Create ConvNet model
    if kwargs['model'] == 'cnn':
        model = ConvNet(n_ts=kwargs['window_size'], 
                        n_features=kwargs['n_features'],
                        n_channels=1,
                        out_dim=len(kwargs['quantiles']),
                        n_filters=kwargs['n_filters'],
                        dropout_p=kwargs['dropout'])
    elif kwargs['model'] == 'lstm':
        model = LSTM(input_dim=kwargs['n_features'],
                     hidden_dim=kwargs['hidden_dim'],
                     lstm_dim=kwargs['lstm_dim'],
                     output_dim=len(kwargs['quantiles']),
                     n_layers=kwargs['n_layers'],
                     dropout=kwargs['dropout'])

    # Compile the model with the Adam optimizer and custom loss function
    model.compile(optimizer=tf.keras.optimizers.Adam(beta_1=0.89, 
                                                     beta_2=0.99, 
                                                     weight_decay=0.02),
                                                     loss=tilted_loss)

    # Create a callback to stop training early if val_loss does not improve
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=10)

    # Train the model
    print(X_train.shape, y_train.shape)
    hist_ = model.fit(X_train, y_train, batch_size=batch_size, 
                      shuffle=True, validation_split=0.3, 
                      epochs=n_epoch, callbacks=[callback])



    # Evaluate on train data
    (nse_train, nnse_train, fig_train), confidence_score = evaluate_preds(model, prod_store,
                                                      train_ds, batch_size,
                                                      y_mu, y_sigma,
                                                      q_in=kwargs['q_in'],
                                                      quantiles=kwargs['quantiles'])
    
    print(f"Train NSE: {nse_train:.3f}")
    print(f"Train Normalized NSE: {nnse_train:.3f}")
    print(f"Confidence Score: {confidence_score:.3f}")
   
    fig_train.savefig(os.path.join(plot_dir, f"{station_id}_train.png"))
    
    # Evaluate on val data
    (nse_val, nnse_val, fig_val), confidence_score_val = evaluate_preds(model, prod_store,
                                                val_ds, batch_size,
                                                y_mu, y_sigma,
                                                q_in=kwargs['q_in'],
                                                quantiles=kwargs['quantiles'])
    print(f"Validation NSE: {nse_val:.3f}")
    print(f"Validation Normalized NSE: {nnse_val:.3f}")
    print(f"Validation Confidence Score: {confidence_score_val:.3f}")
   
    fig_val.savefig(os.path.join(plot_dir, f"{station_id}_val.png"))

    # Write results to file
    dikt = {
        'station_id': station_id,
        'x1': x1,
        'nse_train': nse_train,
        'nnse_train': nnse_train,
        'confidence_score': confidence_score,
        'nse_val': nse_val,
        'nnse_val': nnse_val,
        'confidence_score_val': confidence_score_val,
        'run_ts': dt.datetime.now()
    }
    df = pd.DataFrame(dikt, index=[0])

    csv_path = os.path.join(run_dir, 'result.csv')
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)

    return nse_train, nse_val
















# ------------
if __name__ == '__main__':
    # Parse command line arguments
    args = parser.parse_args()

    if args.q_in:
        args.n_features += 1

    # Create Directories
    if not os.path.exists(args.run_dir):
        os.makedirs(args.run_dir)

    plot_dir = os.path.join(args.run_dir, 'plots')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    with open(os.path.join(args.run_dir, 'run_params.json'), 'w') as f_args:
        json.dump(vars(args), f_args, indent=2)


    print(args)

    if args.station_id is None:

        station_ids = get_station_list(args.data_dir, args.sub_dir)
        
        for ind, station_id in enumerate(station_ids):
            print(f"\n{ind+1}/{len(station_ids)}: Reading data for station_id: {station_id}\n")
            args.station_id = station_id
            train_ds, val_ds = read_dataset_from_file(
                                                        args.data_dir, 
                                                        args.sub_dir, 
                                                        station_id=station_id
                                                    )
            print("Training the neural network model..")
            nse_train, nse_val = train_and_evaluate(
                                    train_ds, val_ds, 
                                    **vars(args)
                                )

    else:
        print(f"Reading data for station_id: {args.station_id}")
        train_ds, val_ds = read_dataset_from_file(
                                                    args.data_dir, 
                                                    args.sub_dir, 
                                                    station_id=args.station_id
                                                )
        print("Training the neural network model..")
        nse_train, nse_val = train_and_evaluate(
                                train_ds, val_ds,
                                **vars(args)
                            )

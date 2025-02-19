import argparse
import datetime as dt
import json
import os
import sys

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torchdata

from tqdm import tqdm
from scipy.stats import genextreme

sys.path.append("../")

from model.ml import ConvNet, LSTM, ConvNetAE
from model.hydro.gr4j_prod import ProductionStorage
from model.utils.training import EarlyStopper
from model.utils.evaluation import evaluate
from data.utils import read_dataset_from_file, get_station_list
from camels_aus.repository import CamelsAus

# Create parser
parser = argparse.ArgumentParser(description="Train Quantile DeepGR4J model")

parser.add_argument('--data-dir', type=str, default='/data/camels/aus/')
parser.add_argument('--sub-dir', type=str, required=True)
parser.add_argument('--station-id', type=str, default=None)
parser.add_argument('--run-dir', type=str, default='/project/results/qdeepgr4j')
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--n-epoch', type=int, default=300)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight-decay', type=float, default=0.02)
parser.add_argument('--model', type=str, default='cnn')
# parser.add_argument('--n-filters', nargs='+', type=int)
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
parser.add_argument('--forecast-horizon', type=int, default=5)


# ----
def tilted_loss(q, y, f):
    e = (y-f)
    return torch.mean(torch.maximum(q*e, (q-1)*e))

def val_step(model, dl, quantiles):
    total_loss = 0.
    model.eval()
    for j, (t, X, y) in enumerate(dl, start=1):
        y_hat = model(X)
        batch_loss = 0.
        for j, q in enumerate(quantiles):
            batch_loss += tilted_loss(q, y[:, :, 0], y_hat[:, :, j])
        total_loss += batch_loss
    return (total_loss/j).detach()


def train_step(model, dl, quantiles, opt):
    total_loss = 0.
    model.train()
    for i, (t, X, y) in enumerate(dl, start=1):
        opt.zero_grad()
        y_hat = model(X)
        batch_loss = 0.
        for j, q in enumerate(quantiles):
            batch_loss += tilted_loss(q, y[:, :, 0], y_hat[:, :, j])
        total_loss += batch_loss
        batch_loss.backward()
        opt.step()
    return (total_loss/i).detach()


def out_of_interval(targets, predictions, low_idx=0, high_idx=None):
    assert (len(predictions.shape)==3), "Expect 3 dimensions in predictions"
    if high_idx is None:
        high_idx = predictions.shape[-1] - 1
    out_of_interval = (targets < predictions[:, :, low_idx]) | (targets > predictions[:, :, high_idx])
    return out_of_interval.sum()/out_of_interval.shape[0]


def evaluate_preds(model, prod_store, ds, batch_size, y_mu, y_sigma, q_in, quantiles=[0.5], threshold=0.0):
    # Evaluate on train data
    model.eval()
    dl = torchdata.DataLoader(ds, 
                              batch_size=batch_size,
                              shuffle=False)

    # Empty list to store batch-wise tensors
    P = []
    ET = []
    Q = []
    Q_hat = []
    T = []

    for i, (t, X, y) in enumerate(dl, start=1):
        
        y_hat = model(X)

        T.append(t.detach().numpy())
        Q.append((y*y_sigma+y_mu).detach().numpy())
        Q_hat.append((y_hat*y_sigma+y_mu).detach().numpy())
        
        if 'conv' in model.__class__.__name__.lower():
            if q_in:
                X_inv = X[:, 0, -1, :-1]*prod_store.sigma+prod_store.mu
            else:
                X_inv = X[:, 0, -1]*prod_store.sigma+prod_store.mu
        elif 'lstm' in model.__class__.__name__.lower():
            if q_in:
                X_inv = X[:, -1, :-1]*prod_store.sigma+prod_store.mu
            else:
                X_inv = X[:, -1]*prod_store.sigma+prod_store.mu
        
        P.append((X_inv[:, 0]).detach().numpy())
        ET.append((X_inv[:, 1]).detach().numpy())
    
    P = np.concatenate(P, axis=0)
    ET = np.concatenate(ET, axis=0)
    Q = np.concatenate(Q, axis=0).squeeze(axis=-1)
    Q_hat = np.clip(np.concatenate(Q_hat, axis=0), 0, None)
    T = np.concatenate(T, axis=0)

    print(f"Shape of T: {T.shape}")

    evaluation_metrics = evaluate(P, ET, T[:, 0], Q[:, 0], Q_hat[:, 0],
                                  quantiles=quantiles,
                                  threshold=threshold)
    
    interval_evaluation = out_of_interval(targets=Q, predictions=Q_hat)

    return evaluation_metrics, interval_evaluation


def create_sequence(t, X, y, window_size, q_in, forecast_horizon):

        assert window_size is not None, "Window size cannot be NoneType."

        # Create empyty sequences
        ts, Xs, ys = [], [], []

        if q_in:
            # Add sequences to Xs and ys
            for i in range(1, len(X) - window_size - forecast_horizon + 1):
                ts.append(t[i + window_size - 1: i + window_size - 1 + forecast_horizon])
                Xs.append(torch.concat([
                                        X[i: (i + window_size)], 
                                        y[i-1: (i + window_size - 1)]
                                    ], dim=1)
                        )
                ys.append(y[i + window_size - 1: i + window_size - 1 + forecast_horizon])
        else:
            # Add sequences to Xs and ys
            for i in range(len(X) - window_size - forecast_horizon + 1):
                ts.append(t[i + window_size: i + window_size + forecast_horizon])
                Xs.append(X[i: (i + window_size)])
                ys.append(y[i + window_size: i + window_size + forecast_horizon])

        ts, Xs, ys = torch.stack(ts), torch.stack(Xs), torch.stack(ys)
        
        if 'cnn' in args.model:
            Xs = torch.unsqueeze(Xs, dim=1)

        return ts, Xs, ys

def get_dataloaders(train_ds, val_ds, gr4j_run_dir, station_id, batch_size, **kwargs):
    # Get tensors from dataset
    t_train, X_train, y_train = train_ds.tensors
    t_val, X_val, y_val = val_ds.tensors

    # Handle nan values
    X_train = torch.nan_to_num(X_train)
    X_val = torch.nan_to_num(X_val)

    # Mean and std 
    y_mu = y_train.mean(dim=0)
    y_sigma = y_train.std(dim=0)

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
    t_train, X_train, y_train = create_sequence(t_train, inp_train, y_train, 
                                        window_size=kwargs['window_size'],
                                        q_in=kwargs['q_in'],
                                        forecast_horizon=kwargs['forecast_horizon'])
    t_val, X_val, y_val = create_sequence(t_val, inp_val, y_val, 
                                    window_size=kwargs['window_size'], 
                                    q_in=kwargs['q_in'],
                                    forecast_horizon=kwargs['forecast_horizon'])

    # Create Sequence Datasets and DataLoaders
    train_ds = torchdata.TensorDataset(t_train, X_train, y_train)
    train_dl = torchdata.DataLoader(train_ds, 
                                    batch_size=batch_size, 
                                    shuffle=True)

    val_ds = torchdata.TensorDataset(t_val, X_val, y_val)
    val_dl = torchdata.DataLoader(val_ds, 
                                    batch_size=batch_size,
                                    shuffle=True)
    
    return train_ds, val_ds, train_dl, val_dl, y_mu, y_sigma, prod_store, x1


def compute_flooding_threshold(repo, station_id):
    # Flooding threshold
    streamflow_data = repo.daily_data[['streamflow_mmd', 'streamflow_MLd_infilled']].sel(time=slice(dt.datetime(1980, 1, 1), dt.datetime(2005, 1, 1)))
    streamflow_data = streamflow_data.sel(station_id=station_id)
    streamflow_data = streamflow_data.to_dataframe()

    # Extract annual maximum streamflow
    annual_max = streamflow_data.resample("Y").max()

    # Fit GEV distribution
    shape, loc, scale = genextreme.fit(annual_max.streamflow_mmd.dropna())
    flooding_threshold = genextreme.ppf(0.8, shape, loc, scale)  # 10-year flood threshold
    print(f"Estimated 5-year flood discharge: {flooding_threshold:.2f} mm/day")
    
    return flooding_threshold


def train_model(model, train_dl, val_dl, opt, early_stopper, n_epoch, quantiles):
    pbar = tqdm(range(1, n_epoch+1))

    for epoch in pbar:
        # Train step
        train_loss = train_step(model, train_dl, quantiles, opt)

        # Validation step
        val_loss = val_step(model, val_dl, quantiles)
        
        pbar.set_description(f"""Epoch {epoch} loss: {train_loss.numpy():.4f} val_loss: {val_loss.numpy():.4f}""")

        if early_stopper.early_stop(val_loss):
            break
    
    return model


def train_and_evaluate(train_ds, val_ds,
                        repo,
                        station_id, n_epoch=100, 
                        batch_size=256, lr=0.001,
                        run_dir='/project/results/qdeepgr4j_cnn',
                        gr4j_run_dir='/project/results/gr4j',
                        **kwargs):

    # Replace the placeholder with the function call
    res = get_dataloaders(train_ds, val_ds, 
                          gr4j_run_dir, station_id, 
                          batch_size, **kwargs)
    
    train_ds, val_ds, train_dl, val_dl, y_mu, y_sigma, prod_store, x1 = res

    # Create ConvNet model
    if kwargs['model'] == 'cnn':
        model = ConvNetAE(ts_in=kwargs['window_size'],
                            in_dim=kwargs['n_features'],
                            ts_out=kwargs['forecast_horizon'],
                            out_dim=len(kwargs['quantiles']),
                            hidden_dim=kwargs['hidden_dim'])
        
    elif kwargs['model'] == 'lstm':
        model = LSTM(input_dim=kwargs['n_features'],
                    hidden_dim=kwargs['hidden_dim'],
                    lstm_dim=kwargs['lstm_dim'],
                    output_dim=len(kwargs['quantiles']),
                    n_layers=kwargs['n_layers'],
                    dropout=kwargs['dropout'])
    
    # Create optimizer and loss instance
    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=kwargs['weight_decay'],
                           betas=(0.89, 0.97))

    # Early stopping
    early_stopper = EarlyStopper(patience=10, min_delta=0.01)

    # Train model
    model = train_model(model, train_dl, val_dl, opt, 
                        early_stopper, n_epoch, kwargs['quantiles'])

    # Compute flooding threshold
    flooding_threshold = compute_flooding_threshold(repo, station_id)

    print(len(train_ds.tensors), len(val_ds.tensors))

    # Evaluate on train data
    (nse_train, nnse_train, fig_train), confidence_score = evaluate_preds(model, prod_store,
                                                      train_ds, batch_size,
                                                      y_mu, y_sigma,
                                                      q_in=kwargs['q_in'],
                                                      quantiles=kwargs['quantiles'],
                                                      threshold=flooding_threshold)
    
    print(f"Train NSE: {nse_train:.3f}")
    print(f"Train Normalized NSE: {nnse_train:.3f}")
    print(f"Confidence Score: {confidence_score:.3f}")
   
    fig_train.savefig(os.path.join(plot_dir, f"{station_id}_train.png"))
    
    # Evaluate on val data
    (nse_val, nnse_val, fig_val), confidence_score_val = evaluate_preds(model, prod_store,
                                                val_ds, batch_size,
                                                y_mu, y_sigma,
                                                q_in=kwargs['q_in'],
                                                quantiles=kwargs['quantiles'],
                                                threshold=flooding_threshold)
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

    # Load streamflow data
    data_dir = '../../data/camels/aus'
    repo = CamelsAus()
    repo.load_from_text_files(data_dir)

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
                                    repo,
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
                                repo,
                                **vars(args)
                            )

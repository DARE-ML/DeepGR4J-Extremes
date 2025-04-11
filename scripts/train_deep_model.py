import argparse
import datetime as dt
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torchdata
from torchsummary import summary


from tqdm import tqdm
from scipy.stats import genextreme

sys.path.append("../")
plt.rcParams.update({'font.size': 22})

from model.ml import ConvNetAE, MultiStepLSTMNet, LSTMNet, RNN, MLP
from model.utils.training import EarlyStopper
from model.utils.evaluation import evaluate, interval_score, plot_quantiles
from data.utils import (read_dataset_from_file, \
                       get_station_list, \
                       get_station_list_from_state)
from camels_aus.repository import CamelsAus

# Create parser
parser = argparse.ArgumentParser(description="Train Quantile DeepGR4J model")

parser.add_argument('--data-dir', type=str, default='../../../data/camels/aus/')
parser.add_argument('--sub-dir', type=str, required=True)
parser.add_argument('--station-id', type=str, default=None)
parser.add_argument('--state', type=str, default=None)
parser.add_argument('--run-dir', type=str, default='/project/results/qdeepgr4j')
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--n-epoch', type=int, default=300)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight-decay', type=float, default=0.02)
parser.add_argument('--model', type=str, default='cnn')
# parser.add_argument('--n-filters', nargs='+', type=int)
parser.add_argument('--input-dim', type=int, default=5)
parser.add_argument('--hidden-dim', type=int, default=32)
parser.add_argument('--lstm-dim', type=int, default=64)
parser.add_argument('--n-layers', type=int, default=2)
parser.add_argument('--quantiles', nargs='+', type=float)
parser.add_argument('--n-features', type=int, default=5)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--gr4j-run-dir', type=str, default='/project/results/gr4j')
parser.add_argument('--window-size', type=int, default=7)
parser.add_argument('--q-in', action='store_true')
parser.add_argument('--forecast-horizon', type=int, default=5)
parser.add_argument('--num-runs', type=int, default=1)
parser.add_argument('--n-stations', type=int, default=None)


# Add device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----
def tilted_loss(q, y, f):
    e = (y-f)
    if q == 0.5:
        return torch.mean(torch.abs(e))
    return torch.mean(torch.maximum(q*e, (q-1)*e))


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

        ts, Xs, ys = torch.stack(ts).to(device), torch.stack(Xs).to(device), torch.stack(ys).to(device)
        
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
    train_len = len(t_train)
    X = torch.cat([X_train, X_val], dim=0)

    # Scale input features
    X_mu = X.mean(dim=0)
    X_sigma = X.std(dim=0)
    X = (X - X_mu)/X_sigma

    # Split data into train and val
    inp_train = X[:train_len]
    inp_val = X[train_len:]

    # Create Input sequence
    t_train, X_train, y_train = create_sequence(t_train, inp_train, y_train, 
                                        window_size=kwargs['window_size'],
                                        q_in=kwargs['q_in'],
                                        forecast_horizon=kwargs['forecast_horizon'])
    t_val, X_val, y_val = create_sequence(t_val, inp_val, y_val, 
                                    window_size=kwargs['window_size'], 
                                    q_in=kwargs['q_in'],
                                    forecast_horizon=kwargs['forecast_horizon'])

    t_train, X_train, y_train = t_train.to(device), X_train.to(device), y_train.to(device)
    t_val, X_val, y_val = t_val.to(device), X_val.to(device), y_val.to(device)

    # Create Sequence Datasets and DataLoaders
    train_ds = torchdata.TensorDataset(t_train, X_train, y_train)
    train_dl = torchdata.DataLoader(train_ds, 
                                    batch_size=batch_size, 
                                    shuffle=True)

    val_ds = torchdata.TensorDataset(t_val, X_val, y_val)
    val_dl = torchdata.DataLoader(val_ds, 
                                    batch_size=batch_size,
                                    shuffle=True) 
    
    return train_ds, val_ds, train_dl, val_dl, X_mu, X_sigma, y_mu, y_sigma, x1


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


def evaluate_preds(models, ds, batch_size, X_mu, X_sigma, y_mu, y_sigma, q_in, quantiles=[0.5], threshold=0.0, run_dir=None, dataset='train'):

    # Move models to device
    models = [model.to(device) for model in models]

    # Evaluate on train data
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

        t, X, y = t.to(device), X.to(device), y.to(device)

        y_hat = []
        
        for model in models:
            model.eval()
        
            y_hat.append(model(X))
        
        y_hat = torch.concat(y_hat, dim=-1)

        T.append(t.detach().cpu().numpy())
        Q.append((y*y_sigma+y_mu).detach().cpu().numpy())
        Q_hat.append((y_hat*y_sigma+y_mu).detach().cpu().numpy())
        
        if 'conv' in model.__class__.__name__.lower():
            if q_in:
                X_inv = X[:, 0, -1, :-1]*X_sigma + X_mu
            else:
                X_inv = X[:, 0, -1]*X_sigma+X_mu
        else:
            if q_in:
                X_inv = X[:, -1, :-1]*X_sigma+X_mu
            else:
                X_inv = X[:, -1]*X_sigma+X_mu
        
        P.append((X_inv[:, 0]).detach().cpu().numpy())
        ET.append((X_inv[:, 1]).detach().cpu().numpy())
    
    P = np.concatenate(P, axis=0)
    ET = np.concatenate(ET, axis=0)
    Q = np.concatenate(Q, axis=0).squeeze(axis=-1)
    Q_hat = np.clip(np.concatenate(Q_hat, axis=0), 0, None)
    T = np.concatenate(T, axis=0)

    # Plot predictions
    forecast_horizon = Q.shape[1]
    fig, ax = plt.subplots(forecast_horizon, 1,
                           figsize=(16, 6*forecast_horizon),
                           sharex=True)

    # Empty lists to store evaluation metrics
    evaluation_metrics = [None]*forecast_horizon
    interval_metrics = [None]*forecast_horizon

    for k in range(forecast_horizon):

        print(f"Evaluating timestep {k+1}...")

        # Evaluate prediction metrics
        evaluation_metrics[k] = evaluate(Q[:, k], Q_hat[:, k],
                                      quantiles=quantiles)
        
        # Evaluate interval metrics
        interval_metrics[k] = interval_score(targets=Q[:, k],
                                           predictions=Q_hat[:, k])

        # Plot quantiles
        fig = plot_quantiles(T[:, k], Q[:, k], Q_hat[:, k],
                               quantiles, threshold, ax=ax[k])
        print(dt.datetime.fromtimestamp(T.min()), 
              dt.datetime.fromtimestamp(T.max()))
        
        # Set title
        ax[k].set_title(f"Streamflow prediction at timestep {k+1}")
        ax[k].legend()
    
    # Save plot to file
    ax[-1].set_xlabel("Time")
    ax[-1].legend()
    fig.savefig(os.path.join(run_dir, f"{dataset}_predictions.png"),
                bbox_inches='tight')
    
    # Format evaluation metrics
    evaluation_metrics = np.array(evaluation_metrics).T
    interval_metrics = np.array(interval_metrics)

    # Flood risk indicator
    T_time = np.array(list(map(dt.datetime.fromtimestamp, T[:, 0])))
    T_time = T_time[-365*2:]
    Q_hat = Q_hat[-365*2:]
    Q = Q[-365*2:]
    alert = np.array([-1]*len(T_time))
    predicted_quantiles = Q_hat.max(axis=1)

    # Assign alert levels
    alert[predicted_quantiles[:, -1] > threshold] = 0
    alert[predicted_quantiles[:, int(len(quantiles)//2)] > threshold] = 1
    alert[predicted_quantiles[:, 0] > threshold] = 2

    # Plot flood risk indicator
    fig, ax = plt.subplots(3, 1, figsize=(16, 14), sharex=True,
                           gridspec_kw={'height_ratios': [2, 2, 1]})

    # Plot streamflow
    ax[0].plot(T_time, Q, color='black', alpha=0.75)
    ax[0].axhline(y=threshold, color='blue', linestyle='--',
                  label='Flooding Threshold')
    ax[0].set_ylabel("Flow (mm/day)")
    ax[0].set_title("Observed Streamflow")
    ax[0].grid(True)
    ax[0].legend()

    # Plot streamflow quantile predictions
    ax[1].plot(T_time, predicted_quantiles[:, 1], 
               color='red', alpha=0.75, label='Predicted')
    ax[1].fill_between(T_time, predicted_quantiles[:, 0], 
                       predicted_quantiles[:, -1], alpha=0.5, color='green')
    ax[1].axhline(y=threshold, color='blue', linestyle='--',
                  label='Flooding Threshold')
    ax[1].set_ylabel("Flow (mm/day)")
    ax[1].set_title("Predicted Streamflow Quantiles")
    ax[1].grid(True)
    ax[1].set_ylim(0, max(Q.max(), threshold))

    # Plot alert levels
    ax[2].scatter(T_time[alert==2], alert[alert==2], 
                   c='red', s=35, marker='x',)
    ax[2].scatter(T_time[alert==1], alert[alert==1], 
                   c='darkorange', s=30, marker='x')
    ax[2].scatter(T_time[alert==0], alert[alert==0], 
                   c='gold', s=30, marker='x')
    ax[2].scatter(T_time[alert==-1], alert[alert==-1], 
                   c='green', s=5, marker='_')
    ax[2].set_title("Flood Risk Indicator")
    ax[2].set_xlabel("Time")
    ax[2].set_ylabel("Alert Level")

    # Configure y-ticks
    y = [-1, 0, 1, 2]
    ax[2].set_yticks(y)

    remap = {-1: "Unlikely", 0: "Low", 1: "Moderate", 2: "High"}
    ax[2].set_yticklabels(map(lambda yy: remap[yy] if yy in remap else '' , y))
    
    ax[2].grid(True)

    # Save plot to file
    fig.savefig(os.path.join(run_dir, f"{dataset}_flood_risk_indicator.png"),
                bbox_inches='tight')
    
    return evaluation_metrics, fig, interval_metrics


def val_step(model, dl, quantiles):
    total_loss = 0.
    model.eval()
    for i, (t, X, y) in enumerate(dl, start=1):
        t, X, y = t.to(device), X.to(device), y.to(device)
        y_hat = model(X)
        batch_loss = 0.
        for j, q in enumerate(quantiles):
            batch_loss += tilted_loss(q, y[:, :, 0], y_hat[:, :, j])
        total_loss += batch_loss
    return (total_loss/i).detach()


def train_step(model, dl, quantiles, opt):
    total_loss = 0.
    model.train()
    for i, (t, X, y) in enumerate(dl, start=1):
        t, X, y = t.to(device), X.to(device), y.to(device)
        opt.zero_grad()
        y_hat = model(X)
        batch_loss = 0.
        for j, q in enumerate(quantiles):
            batch_loss += tilted_loss(q, y[:, :, 0], y_hat[:, :, j])
        total_loss += batch_loss
        batch_loss.backward()
        opt.step()
    return (total_loss/i).detach()


def train_model(model, train_dl, val_dl, opt, early_stopper, n_epoch, quantiles):
    model = model.to(device)
    pbar = tqdm(range(1, n_epoch+1))
    
    train_losses = []
    val_losses = []

    for epoch in pbar:
        # Train step
        train_loss = train_step(model, train_dl, quantiles, opt)
        train_losses.append(train_loss.item())

        # Validation step
        val_loss = val_step(model, val_dl, quantiles)
        val_losses.append(val_loss.item())
        
        pbar.set_description(f"""Epoch {epoch} loss: {train_loss.numpy():.4f} val_loss: {val_loss.numpy():.4f}""")

        if early_stopper.early_stop(val_loss):
            break

    # Plot training and validation losses
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    ax.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Losses')
    ax.legend()
    plt.grid(True)
    
    return model, fig


def convert_results_to_df(mse_score, nse_score, nnse_score, confidence_score, quantiles, forecast_horizon):

    # Convert to dataframe
    res = np.concatenate([mse_score, nse_score, nnse_score], axis=0)
    index = pd.MultiIndex.from_product([['MSE', 'NSE', 'Normalized NSE'], quantiles], 
                                        names=['Metric', 'Quantile'])
    res = pd.DataFrame(res, 
                       columns=[f"Step {t}" for t in range(1, forecast_horizon+1)],
                       index=index)

    # Add confidence scores to the dataframe
    confidence_scores_df = pd.DataFrame(confidence_score.reshape(1, -1),
                                        columns=[f"Step {t}" for t in range(1, forecast_horizon+1)],
                                        index=pd.MultiIndex.from_tuples([('Confidence Score', None)], names=['Metric', 'Quantile']))

    res = pd.concat([res, confidence_scores_df])

    return res


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
    
    train_ds, val_ds, train_dl, val_dl,  X_mu, X_sigma, y_mu, y_sigma, x1 = res

    # Create ConvNet model
    if kwargs['model'] == 'cnn':
        models = [ ConvNetAE(ts_in=kwargs['window_size'],
                            in_dim=kwargs['n_features'],
                            ts_out=kwargs['forecast_horizon'],
                            out_dim=1,
                            hidden_dim=kwargs['hidden_dim']).to(device)
        for _ in range(len(kwargs['quantiles']))]
        

    elif kwargs['model'] == 'lstm':
        # models = [MultiStepLSTMNet(input_dim=kwargs['n_features'],
        #                         hidden_dim=kwargs['hidden_dim'],
        #                         lstm_dim=kwargs['lstm_dim'],
        #                         output_dim=1,
        #                         n_layers=kwargs['n_layers'],
        #                         forecast_horizon=kwargs['forecast_horizon'],
        #                         dropout=kwargs['dropout']).to(device)

        models = [LSTMNet(input_dim=kwargs['n_features'],
                                hidden_dim=kwargs['hidden_dim'],
                                lstm_dim=kwargs['lstm_dim'],
                                output_dim=kwargs['forecast_horizon'],
                                n_layers=kwargs['n_layers'],
                                dropout=kwargs['dropout']).to(device)
                    for _ in range(len(kwargs['quantiles']))]

    elif kwargs['model'] == 'rnn':
        models = [RNN(input_dim=kwargs['n_features'],
                        hidden_dim=kwargs['hidden_dim'],
                        output_dim=kwargs['forecast_horizon'],
                        n_layers=kwargs['n_layers'],
                        dropout=kwargs['dropout']).to(device)
                    for _ in range(len(kwargs['quantiles']))]
        
    elif kwargs['model'] == 'mlp':
        models = [MLP(input_size=kwargs['n_features']*kwargs['window_size'],
                      hidden_sizes=[64, 32, 64, 32, 16],
                      output_size=kwargs['forecast_horizon'],
                      activation=nn.ReLU).to(device)
                    for _ in range(len(kwargs['quantiles']))]

    # Station results directory
    station_dir = os.path.join(run_dir, 'stations', station_id)
    if not os.path.exists(station_dir):
        os.makedirs(station_dir)

    # Quantiles
    quantiles = kwargs['quantiles']
    
    for id, model in enumerate(models):
    
        # Create optimizer and loss instance
        opt = torch.optim.Adam(model.parameters(), lr=lr,
                            weight_decay=kwargs['weight_decay'],
                            betas=(0.89, 0.97))

        # Early stopping
        early_stopper = EarlyStopper(patience=10, min_delta=0.01)


        # Print model summary
        if args.model == 'cnn':
            print(summary(model, input_size=(1, kwargs['window_size'], 
                                        kwargs['n_features'])))
        # elif args.model == 'lstm':
        #     print(summary(model, input_size=(kwargs['window_size'], 
        #                                  kwargs['n_features'])))


        # Train model
        model, loss_plot = train_model(model, train_dl, val_dl, opt, 
                            early_stopper, n_epoch, [quantiles[id]])


        # Save plot
        loss_plot.savefig(os.path.join(station_dir, 
                                       f"loss_{quantiles[id]:.2f}.png"),
                          bbox_inches='tight')

    # Compute flooding threshold
    flooding_threshold = compute_flooding_threshold(repo, station_id)

    # Evaluate on train data
    metrics = evaluate_preds(models, train_ds, batch_size, X_mu, X_sigma,
                             y_mu, y_sigma, q_in=kwargs['q_in'],
                             quantiles=kwargs['quantiles'],
                             threshold=flooding_threshold,
                             run_dir=station_dir, dataset='train')
    
    evaluation_metrics, fig_train, confidence_score = metrics

    # Extract evaluation metrics
    mse_train = evaluation_metrics[:, 0]
    nse_train = evaluation_metrics[:, 1]
    nnse_train = evaluation_metrics[:, 2]

    # Convert results to dataframe
    res_train = convert_results_to_df(mse_train, nse_train, nnse_train, 
                                confidence_score, kwargs['quantiles'], 
                                kwargs['forecast_horizon'])
    
    res_train.loc[:, 'Dataset'] = 'train'

    # Print results for train data
    print(f"Results for station_id: {station_id}")
    print("Train Data:")
    print(res_train)
   
    # # Save plot
    # fig_train.savefig(os.path.join(station_dir, f"train.png"), 
    #                   bbox_inches='tight')
    
    # Evaluate on val data
    metrics = evaluate_preds(models, val_ds, batch_size, X_mu, X_sigma,
                             y_mu, y_sigma, q_in=kwargs['q_in'],
                             quantiles=kwargs['quantiles'],
                             threshold=flooding_threshold,
                             run_dir=station_dir, dataset='val')
    
    evaluation_metrics, fig_val, confidence_score_val = metrics
    
    # Extract evaluation metrics
    mse_val = evaluation_metrics[:, 0]
    nse_val = evaluation_metrics[:, 1]
    nnse_val = evaluation_metrics[:, 2]

    # Convert results to dataframe
    res_val = convert_results_to_df(mse_val, nse_val, nnse_val,
                                confidence_score_val, kwargs['quantiles'],
                                kwargs['forecast_horizon'])
    res_val.loc[:, 'Dataset'] = 'validation'

    # Print results for validation data
    print("Validation Data:")
    print(res_val)
   
    # # Save plot
    # fig_val.savefig(os.path.join(station_dir, "val.png"), bbox_inches='tight')

    # Save results to csv
    res_df = pd.concat([res_train, res_val]).reset_index()
    csv_path = os.path.join(station_dir, 'results.csv')
    res_df.to_csv(csv_path, float_format='%.4f')

    return res_df



if __name__ == '__main__':
    # Parse command line arguments
    args = parser.parse_args()

    if args.q_in:
        args.n_features += 1

    # Load streamflow data
    repo = CamelsAus()
    repo.load_from_text_files(args.data_dir)

    # Get station ids
    if args.station_id is None:
        if args.state is not None:
            args.run_dir = os.path.join(args.run_dir, args.state)
            if not os.path.exists(args.run_dir):
                os.makedirs(args.run_dir)
            station_ids = get_station_list_from_state(args.data_dir, 
                                                      args.sub_dir,
                                                      args.state, 
                                                      args.n_stations)
            station_from_data = get_station_list(args.data_dir, args.sub_dir)
            print(f"Station ids from {args.state} state: {station_ids}")
            station_ids = list(set(station_ids) & set(station_from_data))
            print(f"Selected {len(station_ids)} stations from {args.state} state.")
        else:
            station_ids = get_station_list(args.data_dir, args.sub_dir)
    else:
        station_ids = [args.station_id]

    # Create Directories
    if not os.path.exists(args.run_dir):
        os.makedirs(args.run_dir)

    # Save run parameters
    print(f"""Running Quantile DeepGR4J model \
          with the following parameters:\n{args}""")
    param_path = os.path.join(args.run_dir, 'params.json')
    with open(param_path, 'w') as f_args:
        json.dump(vars(args), f_args, indent=2)


    avg_res_all = []

    for run in range(args.num_runs):

        # Train and evaluate model for each station
        for ind, station_id in enumerate(station_ids):
            
            print(f"\n{ind+1}/{len(station_ids)}: Reading data for station_id: {station_id}\n")
            args.station_id = station_id
            train_ds, val_ds = read_dataset_from_file(
                                                        args.data_dir, 
                                                        args.sub_dir, 
                                                        station_id=station_id
                                                    )
            
            print("Training the neural network model..")
            res_df = train_and_evaluate(train_ds, val_ds,
                                        repo, **vars(args))

            # Compute average results across all time steps
            res_df = res_df.set_index(['Metric', 'Quantile', 'Dataset'])
            station_res = pd.Series(res_df.mean(axis=1), name=f'{station_id}')
            station_res_df = pd.DataFrame(station_res)
            
            if ind == 0:
                all_res = station_res_df
            else:
                all_res = pd.concat([all_res, station_res_df], axis=1)
                
        
        print("Average results across all stations:")
        print(all_res)

        # Save results to csv
        all_res.reset_index().to_csv(os.path.join(args.run_dir, 'results.csv'), float_format='%.4f')
        avg_res = all_res.T.mean().reset_index()
        
        # Pivot table
        avg_res = pd.pivot_table(avg_res, 
                                 index=['Quantile', 'Dataset'], 
                                 columns='Metric').reset_index()
        
        avg_res.reset_index(drop=True, inplace=True)
        avg_res.columns = ['Quantile', 'Dataset', 'MSE', 
                           'NSE', 'Normalized NSE']

        # Append to list
        avg_res_all.append(avg_res)
    

    # Average results
    avg_res = pd.concat(avg_res_all, axis=0)
    avg_res = avg_res.groupby(['Quantile', 'Dataset']).mean().reset_index()
    avg_res.loc[:, 'RMSE'] = np.sqrt(avg_res.MSE)

    # Save average results to csv
    print(avg_res)
    avg_res.to_csv(os.path.join(args.run_dir, 'average_results.csv'), float_format='%.4f')



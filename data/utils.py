import os
import torch
import joblib
import random

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import torch.utils.data as data


def get_station_list(data_dir, sub_dir):
    # Path to data directory
    sub_dir_path = os.path.join(data_dir, '../processed', sub_dir)
    write_dir_train = os.path.join(sub_dir_path, 'datasets/train')
    return os.listdir(write_dir_train)


def get_station_list_from_state(data_dir, sub_dir, state, n_stations=None):
    # Read locations data
    locations_path = os.path.join(data_dir, '02_location_boundary_area',
                                  'location_boundary_area.csv')
    locations_df = pd.read_csv(locations_path)

    # Streamflow signatures
    signatures_path = os.path.join(data_dir, '03_streamflow',
                                   'streamflow_signatures.csv')
    signatures_df = pd.read_csv(signatures_path)

    # Merge dataframes
    merged_df = pd.merge(locations_df, signatures_df, 
                            left_on='station_id', right_on='station_id')
    
    # Filter by state
    merged_df = merged_df.loc[merged_df['state_outlet'] == state, :]
    print(merged_df)

    # Filter by n_stations
    if n_stations is not None:
        merged_df.sort_values('runoff_ratio', ascending=False, inplace=True)
        stations_from_file = get_station_list(data_dir, sub_dir)
        merged_df = merged_df.loc[merged_df['station_id'].isin(stations_from_file), :]
        return merged_df['station_id'].values[:n_stations]

    return merged_df['station_id'].values


def read_dataset_from_file(data_dir, sub_dir, station_id):

    # Paths to read from
    sub_dir_path = os.path.join(data_dir, '../processed', sub_dir)
    write_dir_train = os.path.join(sub_dir_path, 'datasets/train', station_id)
    write_dir_val = os.path.join(sub_dir_path, 'datasets/val', station_id)
    scaler_path = os.path.join(sub_dir_path, 'scalers')

    # Load train data
    X_train = torch.load(os.path.join(write_dir_train, 'X_train.pt'))
    y_train = torch.load(os.path.join(write_dir_train, 'y_train.pt'))
    t_train = torch.load(os.path.join(write_dir_train, 't_train.pt'))

    # Load validation data
    X_val = torch.load(os.path.join(write_dir_val, 'X_val.pt'))
    y_val = torch.load(os.path.join(write_dir_val, 'y_val.pt'))
    t_val = torch.load(os.path.join(write_dir_val, 't_val.pt'))

    # Create datasets
    train_ds = data.TensorDataset(t_train, X_train, y_train)
    val_ds = data.TensorDataset(t_val, X_val, y_val)

    # Define function result
    out = (train_ds, val_ds)

    if os.path.exists(scaler_path):
        # Read scalers
        x_scaler = joblib.load(os.path.join(scaler_path, 'x_scaler.save'))
        y_scaler = joblib.load(os.path.join(scaler_path, 'y_scaler.save'))

        out = out + (x_scaler, y_scaler)

    return out


if __name__=='__main__':

    data_dir = '/data/camels/aus/'
    sub_dir = 'no-scale'
    station_id = '102101A'

    print(read_dataset_from_file(data_dir, sub_dir, station_id=station_id)[0].tensors)
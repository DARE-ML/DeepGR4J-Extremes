from __future__ import absolute_import

import os
import datetime as dt
import time
import numpy as np
import pandas as pd
import pickle

from camels_aus.repository import CamelsAus
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

WINDOW_SIZE = 7


class CamelsAusDataset(object):
    """Class to read Camels dataset from file
    """

    x_col = ['precipitation_AWAP', 'et_morton_actual_SILO',
            'tmax_awap', 'tmin_awap', 'vprp_awap']
    y_col = ['streamflow_mmd']
    coord_col = ['station_id', 'time']

    def __init__(self, data_dir, x_col=None, y_col=None,
                scale:bool=True, create_seq:bool=True, 
                keep_z:bool=True, window_size:int=WINDOW_SIZE):

        # Path to Camels data
        self.data_dir = data_dir

        # Create data repository
        self.repo = CamelsAus()
        self.repo.load_from_text_files(self.data_dir)
        
        # Xarray dataset object
        self.ds = self.repo.daily_data.sel(time=slice(dt.datetime(1980, 1, 1), dt.datetime(2015, 1, 1)))

        # Define x and y columns
        if x_col is not None:
            self.x_col = x_col
        if y_col is not None:
            self.y_col = y_col

        # DS list
        self.ds_store = self.create_datasets(scale, create_seq, window_size=window_size)
    
    def create_sequence(self, t,  X, y, window_size):

        assert window_size is not None, "Window size cannot be NoneType."

        # Create empyty sequences
        ts, Xs, ys = [], [], []

        # Add sequences to Xs and ys
        for i in range(len(X)-window_size):
            Xs.append(X[i: (i + window_size)])
            ys.append(y[i + window_size-1])
            ts.append(t[i + window_size-1])

        ts, Xs, ys = tf.stack(ts), tf.stack(Xs), tf.stack(ys)

        return ts, Xs, ys

    
    def create_datasets(self, scale, create_seq, window_size=None):

        # Store station ids
        self.stations = self.ds.station_id.to_numpy()

        X_list, y_list, coord_list = [], [], []

        for station_id in self.stations:
            station_ds = self.ds.sel(station_id=station_id)
            station_ds = station_ds[self.x_col + self.y_col].where(
                                    lambda x: x[self.y_col[0]].notnull(), 
                                    drop=True
                                )
            for x_col in self.x_col:
                station_ds = station_ds[self.x_col + self.y_col].where(
                                    lambda x: x[x_col].notnull(), 
                                    drop=True
                                )
            station_df = station_ds.to_pandas().reset_index()

            station_df.time = station_df.time.apply(lambda x: time.mktime(x.timetuple()))

            X_list.append(station_df[self.x_col])
            y_list.append(station_df[self.y_col])
            coord_list.append(station_df[self.coord_col])

        
        X = pd.concat(X_list, axis=0).reset_index(drop=True)
        y = pd.concat(y_list, axis=0).reset_index(drop=True)
        coord = pd.concat(coord_list, axis=0).reset_index(drop=True)
        

        # Scaling preference
        self.scale = scale
        if scale:
            self.x_scaler = StandardScaler()
            self.y_scaler = StandardScaler()
        
            # Scale
            X = self.x_scaler.fit_transform(X)
            y = self.y_scaler.fit_transform(y)
        
        else:
            X = X.values
            y = y.values

        ds_store = {}


        for station_id in self.stations:

            indices = coord.index[coord.station_id==station_id]

            indices_train, indices_val = train_test_split(indices, 
                                                          test_size=0.3, 
                                                          shuffle=False)

            X_train, X_val = (
                tf.convert_to_tensor(X[indices_train]), 
                tf.convert_to_tensor(X[indices_val])
            )
            y_train, y_val = (
                tf.convert_to_tensor(y[indices_train]), 
                tf.convert_to_tensor(y[indices_val])
            )
            time_train, time_val = (
                tf.convert_to_tensor(
                    coord.values[indices_train, 1].astype('float')
                ), tf.convert_to_tensor(
                    coord.values[indices_val, 1].astype('float')
                )
            )

            # Create Sequences
            if create_seq:
                time_train, X_train, y_train = self.create_sequence(
                    time_train, X_train, y_train, 
                    window_size=window_size
                )

                time_val, X_val, y_val = self.create_sequence(
                    time_val, X_val, y_val, 
                    window_size=window_size
                )

            ds_store[station_id] = {
                'train': tf.data.Dataset.from_tensor_slices((time_train, X_train, y_train)),
                'val': tf.data.Dataset.from_tensor_slices((time_val, X_val, y_val))
            }

        return ds_store


    def get_dataloader(self, station_id, train=True, batch_size=64, shuffle=False):
        
        if train: 
            return self.ds_store[station_id]['train'].shuffle(shuffle).batch(batch_size)
        else:
            return self.ds_store[station_id]['val'].batch(batch_size)






class CamelsDataset(object):
    ts_vars = ['precipitation_AWAP', 'et_morton_actual_SILO',
                       'tmax_awap', 'tmin_awap']
    location_vars = ['state_outlet', 'map_zone', 'catchment_area']
    streamflow_vars = ['q_mean', 'stream_elas', 'runoff_ratio', 'high_q_freq',
                       'high_q_dur', 'low_q_freq', 'zero_q_freq']
    target_vars = ['streamflow_mmd']
    ts_slice = slice(dt.datetime(1980, 1, 1), dt.datetime(2015, 1, 1))

    def __init__(self, data_dir, ts_vars=None, location_vars=None,
                 streamflow_vars=None, target_vars=None, window_size=WINDOW_SIZE) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.window_size = window_size
        self.compute_flow_cdf = False

        # Reassign columns
        if ts_vars is not None:
            self.ts_vars = ts_vars
        if location_vars is not None:
            self.location_vars = location_vars
        if streamflow_vars is not None:
            self.streamflow_vars = streamflow_vars
        if target_vars is not None:
            self.target_vars = target_vars

        # Load the data
        self.repo = CamelsAus()
        self.repo.load_from_text_files(data_dir)

        # Scalers
        self.ts_scaler = None
        self.static_scaler = None
        self.target_scaler = None
    
    def get_station_list(self, state_outlet=None, map_zone=None):
        if state_outlet is not None and map_zone is not None:
            loc_df = self.repo.location_attributes.to_dataframe()
            loc_df = loc_df[(loc_df['state_outlet']==state_outlet) & (loc_df['map_zone']==map_zone)]
            return loc_df.reset_index().station_id.unique()
        return self.repo.location_attributes.to_dataframe().reset_index().station_id.unique()

    def get_zones(self):
        return self.repo.location_attributes.to_dataframe().reset_index()[['state_outlet', 'map_zone']].drop_duplicates().values


    def load_scalers(self, filepath):
        with open(filepath, 'rb') as f:
            scalers = pickle.load(f)
            self.ts_scaler = scalers['ts_scaler']
            self.static_scaler = scalers['static_scaler']
            self.target_scaler = scalers['target_scaler']
            
    
    def prepare_data(self, station_list=None, state_outlet=None, map_zone=None):

        # Check if flow_cdf is in target_vars
        if 'flow_cdf' in self.target_vars:
            self.target_vars.remove('flow_cdf')
            if 'streamflow_mmd' not in self.target_vars:
                self.target_vars.append('streamflow_mmd')
            self.compute_flow_cdf = True

        # # Timeseries data
        ts_data = self.repo.daily_data.sel(time=self.ts_slice)[self.ts_vars+self.target_vars].to_dataframe().reset_index()
        ts_data = ts_data.dropna()
        self.ts_data = ts_data[self.ts_vars + ['station_id', 'time']]
        self.targets = ts_data[self.target_vars + ['station_id', 'time']]
        
        # # Static data
        location_data = self.repo.location_attributes.to_dataframe()[self.location_vars]
        streamflow_data = self.repo.streamflow_attributes.to_dataframe()[self.streamflow_vars]
        self.static_data = pd.concat([location_data, streamflow_data], axis=1).reset_index()

        # Filter static data by station_list or state_outlet and map_zone
        if station_list is not None:
            self.station_list = station_list
            self.static_data = self.static_data[self.static_data['station_id'].isin(self.station_list)]
            self.ts_data = self.ts_data[self.ts_data['station_id'].isin(self.station_list)]
        elif state_outlet is not None and map_zone is not None:
            self.static_data = self.static_data[(self.static_data['state_outlet']==state_outlet) & (self.static_data['map_zone']==map_zone)]
            self.station_list = self.static_data.station_id.unique()
            self.ts_data = self.ts_data[self.ts_data['station_id'].isin(self.station_list)]
        else:
            raise ValueError('station_list or state_outlet and map_zone must be provided')
        self.static_data.drop(columns=self.location_vars, inplace=True)

        # Sort data
        self.ts_data = self.ts_data.sort_values(['time', 'station_id'])
        self.static_data = self.static_data.sort_values('station_id')
        self.targets = self.targets.sort_values(['time', 'station_id'])

        # Timestamps
        self._ts = self.ts_data.time.unique()

        # Convert time to sin and cos
        date_min = np.min(self._ts)
        year_seconds = 365.2425*24*60*60
        diff_seconds = (self.ts_data['time'] - date_min).dt.total_seconds().values
        self.ts_data['year_sin'] =  np.sin(diff_seconds * (2*np.pi/year_seconds))
        self.ts_data['year_cos'] =  np.cos(diff_seconds * (2*np.pi/year_seconds))

        
        if self.compute_flow_cdf:
            self.targets = self.add_flow_cdf(self.targets)
            self.targets.drop(columns=['streamflow_mmd'], inplace=True)
            self.target_vars.remove('streamflow_mmd')
            if 'flow_cdf' not in self.target_vars:
                self.target_vars.append('flow_cdf')

        # Scalers
        if self.ts_scaler is None:
            self.ts_scaler = StandardScaler()
            self.static_scaler = StandardScaler()
            self.target_scaler = StandardScaler()

            # Scale
            self.ts_data[self.ts_vars] = self.ts_scaler.fit_transform(self.ts_data[self.ts_vars])
            self.static_data[self.streamflow_vars] = self.static_scaler.fit_transform(self.static_data[self.streamflow_vars])
            self.targets[self.target_vars] = self.target_scaler.fit_transform(self.targets[self.target_vars])
        
        else:
            self.ts_data[self.ts_vars] = self.ts_scaler.transform(self.ts_data[self.ts_vars])
            self.static_data[self.streamflow_vars] = self.static_scaler.transform(self.static_data[self.streamflow_vars])
            self.targets[self.target_vars] = self.target_scaler.transform(self.targets[self.target_vars])


    def save_scalers(self, path):
        path = os.path.join(path, 'scalers.pkl')
        with open(path, 'wb') as f:
            pickle.dump({'ts_scaler': self.ts_scaler, 'static_scaler': self.static_scaler, 'target_scaler': self.target_scaler}, f)

        
    
    def add_flow_cdf(self, target_data):
        target_data_updated = []
        for station_id in self.station_list:
            df = target_data[target_data.station_id == station_id].set_index('time')
            if 'streamflow_mmd' in self.target_vars:
                flow_values = df['streamflow_mmd'].dropna().sort_values(ascending=True)
            else:
                flow_values = df.streamflow_mmd.dropna().sort_values(ascending=True)
            flow_cdf = (np.arange(len(flow_values))+1)/(len(flow_values) + 1)
            flow_cdf = pd.Series(flow_cdf, index=flow_values.index)
            df['flow_cdf'] = flow_cdf
            df.reset_index(inplace=True)
            target_data_updated.append(df)
        return pd.concat(target_data_updated)
        

    def create_datasets(self, ts_data, static_data, target_data):
        station_ids = static_data.station_id.unique()
        
        ts_arr = []
        static_arr = []
        target_arr = []
        station_names = []
        
        for station_id in station_ids:
            
            station_ts = ts_data[ts_data['station_id']==station_id].drop(columns=['station_id', 'time'])
            station_static = static_data[static_data['station_id']==station_id].drop(columns=['station_id'])
            station_targets = target_data[target_data['station_id']==station_id].drop(columns=['station_id', 'time'])
            
            station_ts_data, station_targets = self.create_sequences(station_ts.values,
                                                                     station_targets.values,
                                                                     self.window_size)
            station_static_data = np.repeat(station_static.values, station_ts_data.shape[0], axis=0)
            station_names_data = np.repeat(station_id, station_ts_data.shape[0], axis=0)[:, np.newaxis]
            
            ts_arr.append(station_ts_data)
            static_arr.append(station_static_data)
            target_arr.append(station_targets)
            station_names.append(station_names_data)
        
        self.ts_arr = np.nan_to_num(np.concatenate(ts_arr))
        self.static_arr = np.nan_to_num(np.concatenate(static_arr))
        self.target_arr = np.nan_to_num(np.concatenate(target_arr))
        self.station_names = np.concatenate(station_names)

    def create_sequences(self, x, y, window_size):
        sequences = []
        targets = []
        for i in range(window_size, len(x)):
            sequences.append(x[i-window_size:i])
            targets.append(y[i])
        return np.stack(sequences), np.stack(targets)
    
    def get_datasets(self, test_size=0.2):
        # Create the datasets
        self.create_datasets(self.ts_data, self.static_data, self.targets)

        # Lists to store train and test data
        ts_train, ts_test = [], []
        static_train, static_test = [], []
        target_train, target_test = [], []
        station_names_train, station_names_test = [], []

        for i, station_id in enumerate(self.station_list):
            
            # Station indices
            station_idx = self.station_names.flatten() == station_id

            # Station arrays
            station_ts = self.ts_arr[station_idx]
            station_static = self.static_arr[station_idx]
            station_names = self.station_names[station_idx]
            station_target = self.target_arr[station_idx]

            # Split arrays into train & test
            n_records = station_names.shape[0]
            n_records_train = int(n_records*(1-test_size))

            # Train data
            ts_train.append(station_ts[:n_records_train])
            static_train.append(station_static[:n_records_train])
            target_train.append(station_target[:n_records_train])
            station_names_train.append(station_names[:n_records_train])

            # Test data
            ts_test.append(station_ts[n_records_train:])
            static_test.append(station_static[n_records_train:])
            target_test.append(station_target[n_records_train:])
            station_names_test.append(station_names[n_records_train:])

        # Convert train and test data to tensors
        ts_train = tf.convert_to_tensor(np.concatenate(ts_train, axis=0),
                                        dtype=tf.float32, name='timeseries_train')
        ts_test = tf.convert_to_tensor(np.concatenate(ts_test, axis=0),
                                       dtype=tf.float32, name='timeseries_test')
        static_train = tf.convert_to_tensor(np.concatenate(static_train, axis=0),
                                            dtype=tf.float32, name='static_train')
        static_test = tf.convert_to_tensor(np.concatenate(static_test, axis=0),
                                           dtype=tf.float32, name='static_test')
        target_train = tf.convert_to_tensor(np.concatenate(target_train, axis=0),
                                            dtype=tf.float32, name='target_train')
        target_test = tf.convert_to_tensor(np.concatenate(target_test, axis=0),
                                           dtype=tf.float32, name='target_test')
        station_names_train = tf.convert_to_tensor(np.concatenate(station_names_train, axis=0),
                                                    dtype=tf.string, name='station_names_train')
        station_names_test = tf.convert_to_tensor(np.concatenate(station_names_test, axis=0),
                                                    dtype=tf.string, name='station_names_test')
        
        # Create the datasets
        train_dataset = tf.data.Dataset.from_tensor_slices({'station_id': station_names_train,
                                                            'timeseries': ts_train,
                                                            'static': static_train,
                                                            'target': target_train})

        test_dataset = tf.data.Dataset.from_tensor_slices({'station_id': station_names_test,
                                                           'timeseries': ts_test,
                                                           'static': static_test,
                                                           'target': target_test})
        
        return train_dataset, test_dataset





class HybridDataset(CamelsDataset):

    ts_vars = ['precipitation_AWAP', 'et_morton_actual_SILO',
                       'tmax_awap', 'tmin_awap']
    location_vars = ['state_outlet', 'map_zone']
    streamflow_vars = ['q_mean', 'stream_elas', 'runoff_ratio', 'high_q_freq',
                       'high_q_dur', 'low_q_freq', 'zero_q_freq']
    target_vars = ['streamflow_mmd']
    ts_slice = slice(dt.datetime(1980, 1, 1), dt.datetime(2015, 1, 1))

    def __init__(self, data_dir, gr4j_logfile, prod, window_size=WINDOW_SIZE, **kwargs) -> None:
        super().__init__(data_dir, window_size=window_size, **kwargs)
        self.gr4j_logs = pd.read_csv(gr4j_logfile)
        self.prod = prod
    
    def create_datasets(self, ts_data, static_data, target_data):
        station_ids = static_data.station_id.unique()
        
        ts_arr = []
        static_arr = []
        target_arr = []
        station_names = []
        
        for station_id in station_ids:
            
            station_ts = ts_data[ts_data['station_id']==station_id].drop(columns=['station_id', 'time']).values
            station_static = static_data[static_data['station_id']==station_id].drop(columns=['station_id']).values
            station_targets = target_data[target_data['station_id']==station_id].drop(columns=['station_id', 'time']).values

            # Initialize GR4J Production storage
            x1_param = self.gr4j_logs.loc[self.gr4j_logs['station_id']==station_id, 'x1'].values[0]
            self.prod.set_x1(x1_param)
            station_hybrid_feat = self.prod(tf.convert_to_tensor(station_ts), include_x=False, scale=True)[0].numpy()
            station_ts = np.concatenate([station_ts, station_hybrid_feat], axis=1)
            
            station_ts_data, station_targets = self.create_sequences(station_ts, station_targets, self.window_size)
            station_static_data = np.repeat(station_static, station_ts_data.shape[0], axis=0)
            station_names_data = np.repeat([station_id], station_ts_data.shape[0], axis=0)[:, np.newaxis]
            
            ts_arr.append(station_ts_data)
            static_arr.append(station_static_data)
            target_arr.append(station_targets)
            station_names.append(station_names_data)
        
        self.ts_arr = np.concatenate(ts_arr)
        self.static_arr = np.concatenate(static_arr)
        self.target_arr = np.concatenate(target_arr)
        self.station_names = np.concatenate(station_names)
from __future__ import absolute_import

import datetime as dt
import time
import numpy as np
import pandas as pd
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

    def __init__(self, data_dir, state_outlet=None, map_zone=None,
                 station_list=None, ts_vars=None, location_vars=None,
                 streamflow_vars=None, target_vars=None, window_size=WINDOW_SIZE) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.window_size = window_size
        compute_flow_cdf = False

        # Reassign columns
        if ts_vars is not None:
            self.ts_vars = ts_vars
        if location_vars is not None:
            self.location_vars = location_vars
        if streamflow_vars is not None:
            self.streamflow_vars = streamflow_vars
        if target_vars is not None:
            self.target_vars = target_vars
        
        if 'flow_cdf' in self.target_vars:
            self.target_vars.remove('flow_cdf')
            if 'streamflow_MLd_infilled' not in self.target_vars:
                self.target_vars.append('streamflow_MLd_infilled')
            compute_flow_cdf = True

        # Load the data
        self.repo = CamelsAus()
        self.repo.load_from_text_files(data_dir)

        # # Timeseries data
        ts_data = self.repo.daily_data.sel(time=self.ts_slice)[self.ts_vars+self.target_vars].to_dataframe().reset_index()
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

        # Timestamps
        self._ts = self.ts_data.time.unique()

        # Sort data
        self.ts_data = self.ts_data.sort_values(['time', 'station_id'])
        self.static_data = self.static_data.sort_values('station_id')
        self.targets = self.targets.sort_values(['time', 'station_id'])

        if compute_flow_cdf:
            self.targets = self.add_flow_cdf(self.targets)
            self.targets.drop(columns=['streamflow_MLd_infilled'], inplace=True)
    
    def add_flow_cdf(self, target_data):
        target_data_updated = []
        for station_id in self.station_list:
            df = target_data[target_data.station_id == station_id].set_index('time')
            if 'streamflow_MLd_infilled' in self.target_vars:
                flow_values = df.streamflow_MLd_infilled.dropna().sort_values(ascending=True)
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
        
        self.ts_arr = np.concatenate(ts_arr)
        self.static_arr = np.concatenate(static_arr)
        self.target_arr = np.concatenate(target_arr)
        self.station_names = np.concatenate(station_names)

    def create_sequences(self, x, y, window_size):
        sequences = []
        targets = []
        for i in range(window_size, len(x)):
            sequences.append(x[i-window_size:i])
            targets.append(y[i])
        return np.stack(sequences), np.stack(targets)
    
    def get_datasets(self, test_size=0.2, batch_size=32):
        # Create the datasets
        self.create_datasets(self.ts_data, self.static_data, self.targets)

        # Split the data
        n_stations = np.unique(self.station_names).shape[0]
        n_records = self.station_names.shape[0]/n_stations
        n_records_train = int(n_records*(1-test_size)) * n_stations

        ts_train = self.ts_arr[:n_records_train]
        ts_test = self.ts_arr[n_records_train:]
        static_train = self.static_arr[:n_records_train]
        static_test = self.static_arr[n_records_train:]
        target_train = self.target_arr[:n_records_train]
        target_test = self.target_arr[n_records_train:]
        station_names_train = self.station_names[:n_records_train]
        station_names_test = self.station_names[n_records_train:]

        # Conver train and test data to tensors
        ts_train = tf.convert_to_tensor(ts_train, dtype=tf.float32, name='timeseries_train')
        ts_test = tf.convert_to_tensor(ts_test, dtype=tf.float32, name='timeseries_test')
        static_train = tf.convert_to_tensor(static_train, dtype=tf.float32, name='static_train')
        static_test = tf.convert_to_tensor(static_test, dtype=tf.float32, name='static_test')
        target_train = tf.convert_to_tensor(target_train, dtype=tf.float32, name='target_train')
        target_test = tf.convert_to_tensor(target_test, dtype=tf.float32, name='target_test')
        station_names_train = tf.convert_to_tensor(station_names_train, dtype=tf.string, name='station_names_train')
        station_names_test = tf.convert_to_tensor(station_names_test, dtype=tf.string, name='station_names_test')

        # Create the datasets
        train_dataset = tf.data.Dataset.from_tensor_slices({'station_id': station_names_train,
                                                            'timeseries': ts_train,
                                                            'static': static_train,
                                                            'target': target_train})
        train_dataset = train_dataset.shuffle(140000).batch(batch_size)

        test_dataset = tf.data.Dataset.from_tensor_slices({'station_id': station_names_test,
                                                           'timeseries': ts_test,
                                                           'static': static_test,
                                                           'target': target_test})
        test_dataset = test_dataset.batch(batch_size)
        
        return train_dataset, test_dataset





class HybridDataset(CamelsDataset):

    ts_vars = ['precipitation_AWAP', 'et_morton_actual_SILO',
                       'tmax_awap', 'tmin_awap']
    location_vars = ['state_outlet', 'map_zone']
    streamflow_vars = ['q_mean', 'stream_elas', 'runoff_ratio', 'high_q_freq',
                       'high_q_dur', 'low_q_freq', 'zero_q_freq']
    target_vars = ['streamflow_mmd']
    ts_slice = slice(dt.datetime(1980, 1, 1), dt.datetime(2015, 1, 1))

    def __init__(self, data_dir, gr4j_logfile, prod, 
                 state_outlet=None, map_zone=None,
                 station_list=None, window_size=WINDOW_SIZE) -> None:
        super().__init__(data_dir, state_outlet=state_outlet,
                         map_zone=map_zone, station_list=station_list,
                         window_size=window_size)
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
            prod = self.prod.set_x1(x1_param)
            station_hybrid_feat = prod(tf.convert_to_tensor(station_ts), include_x=False, scale=False)[0].numpy()
            station_ts = np.concatenate([station_ts, station_hybrid_feat], axis=1)
            
            station_ts_data, station_targets = self.create_sequences(station_ts, station_targets, WINDOW_SIZE)
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
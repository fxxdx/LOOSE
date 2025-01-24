import sys
import os
from tqdm import tqdm
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

from utils.data_loader import DataLoader
from utils.config import *
from utils.metrics_loader import MetricsLoader
from numpy import *
from sklearn.preprocessing import StandardScaler
import pathlib
from pathlib import Path
def nd_numpy_to_nested(X, name):


	dim0 = X.shape[0]
	dim1 = X.shape[1]
	dfdata = np.ones((dim0, dim1))
	for v1 in range(dim1):
		dfdata[v1] = [X[v0][v1] for v0 in range(dim0)]
	df = pd.DataFrame(dfdata, columns = name)
	return df


def create_tmp_dataset(
	name,
	save_dir,
	data_path,
	metric_path,
	window_size,
	metric,
	window
):
	"""Generates a new dataset from the given dataset. The time series
	in the generated dataset have been divided in windows:lookback+horizon.
	"""

	# Form new dataset's name
	name = '{}_{}'.format(name, window_size)

	metricsloader = MetricsLoader(metric_path)
	metrics_data = metricsloader.read(metric)


	datasets = dataset_names
	# datasets = dataloader.get_dataset_names()
	pbar = tqdm(datasets)
	for name in pbar:
		PD_data = read_data(os.path.join(data_path, name))
		print(name)
		print(PD_data)
		PDdata = PD_data.values
		name_index = name
		# print("---------------------------")
		# print(metrics_data)
		# print("---------------------------")
		metrics_data_im = metrics_data.loc[name_index]


		metrics_data_im = metrics_data_im[forcast_names]
		# print("after ----------metric_data--------: ", metrics_data_im)

		# Split timeseries and compute labels
		ts_list, labels = split_and_compute_labels(PDdata, metrics_data_im, window_size, window)


	# Save new dataset
		for ts, label in tqdm(zip(ts_list, labels), total=len(ts_list), desc='Save dataset'):

			col_names = PD_data.columns.tolist()


			# df_final = nd_numpy_to_nested(ts, col_names)
			if ts.shape[-1]<=8:
				for i in range(8-ts.shape[-1]+1):
					ts = np.concatenate([ts,ts[:,:,0][:,:,np.newaxis]], axis=-1)
					col_names += [col_names[0]]
			col_names += ['label']
			datalabel = np.zeros((ts.shape[0],ts.shape[1]))
			for i in range(ts.shape[0]):
				for j in range(ts.shape[1]):
					datalabel[i,j] = label[i]
			data = np.concatenate((ts, datalabel[:, :, np.newaxis]), axis=-1)


			# label_df = pd.DataFrame(datalabel, columns = ['labels'])
			dfs = [pd.DataFrame(x, columns = col_names) for x in data]

			df_final = pd.concat(dfs, keys=range(len(dfs)))
			print(df_final)

			target_path = Path(save_dir)
			pathlib.Path(target_path).mkdir(parents=True, exist_ok=True)
			df_final.to_csv(os.path.join(save_dir, name), index=False)


def split_and_compute_labels(x, metrics_data, window_size, stride):
	'''Splits the timeseries, computes the labels and returns 
	the segmented timeseries and the new labels.
	'''
	ts_list = []
	labels = []

	scaler = StandardScaler()
	x = scaler.fit_transform(x)

	# Split time series into windows
	# ts_split = split_ts(x, window_size)
	ts_split = split_ts_win(x, window_size, stride)
	# print(ts_split)
	print("-------win com---------------")
	metric_label = metrics_data.idxmin()
	# Save everything to lists
	ts_list.append(ts_split)
	labels.append(np.ones(len(ts_split)) * forcast_names.index(metric_label))

	assert(
		len(ts_list) == len(labels)
	), "Timeseries split and labels computation error, lengths do not match"
			
	return ts_list, labels



def z_normalization(ts, decimals=5):
	if len(set(ts)) == 1:
		ts = ts - np.mean(ts)
	else:
		ts = (ts - np.mean(ts)) / np.std(ts)
	ts = np.around(ts, decimals=decimals)

	# Test normalization
	assert (
			np.around(np.mean(ts), decimals=3) == 0 and np.around(np.std(ts) - 1, decimals=3) == 0
	), "After normalization it should: mean == 0 and std == 1"

	return ts

def split_ts(data, window_size):
	'''
	Split a timeserie into windows according to window_size.
	If the timeserie can not be divided exactly by the window_size
	then the first window will overlap the second.
	'''

	# Compute the modulo
	modulo = data.shape[0] % window_size

	# Compute the number of windows
	k = data[modulo:,:].shape[0] / window_size
	assert(math.ceil(k) == k)

	# Split the timeserie
	data_split = np.split(data[modulo:,:], k)
	if modulo != 0:
		data_split.insert(0, list(data[:window_size,:]))
	data_split = np.asarray(data_split)

	return data_split


def split_ts_win(data, window_size, stride):

    # Compute the number of windows
    num_windows = math.floor((data.shape[0] - window_size + 1) / stride)
    print("--------------------------", stride)
    print("--------------------------",num_windows)
    # Split the timeserie
    data_split = []
    for i in range(num_windows):
        start = i * stride
        end = start + window_size
        window = data[start:end, :]
        data_split.append(window)

    data_split = np.asarray(data_split)

    return data_split

def read_data(path: str, nrows=None) -> pd.DataFrame:
    """
    Read the data file and return DataFrame.
    According to the provided file path, read the data file and return the corresponding DataFrame.
    :param path: The path to the data file.
    :return:  The DataFrame of the content of the data file.
    """
    data = pd.read_csv(path)
    label_exists = "label" in data["cols"].values

    all_points = data.shape[0]

    columns = data.columns

    if columns[0] == "date":
        n_points = data.iloc[:, 2].value_counts().max()
        print(data.iloc[:, 2].value_counts())
    else:
        n_points = data.iloc[:, 1].value_counts().max()
    # print("------------------n_points-------------------------",n_points)
    # print(all_points)
    is_univariate = n_points == all_points

    n_cols = all_points // n_points
    df = pd.DataFrame()

    cols_name = data["cols"].unique()

    if columns[0] == "date" and not is_univariate:
        df["date"] = data.iloc[:n_points, 0]
        col_data = {
            cols_name[j]: data.iloc[j * n_points: (j + 1) * n_points, 1].tolist()
            for j in range(n_cols)
        }
        df = pd.concat([df, pd.DataFrame(col_data)], axis=1)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

    elif columns[0] != "date" and not is_univariate:
        col_data = {
            cols_name[j]: data.iloc[j * n_points: (j + 1) * n_points, 0].tolist()
            for j in range(n_cols)
        }
        df = pd.concat([df, pd.DataFrame(col_data)], axis=1)

    elif columns[0] == "date" and is_univariate:
        df["date"] = data.iloc[:, 0]
        df[cols_name[0]] = data.iloc[:, 1]

        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

    else:
        df[cols_name[0]] = data.iloc[:, 0]

    if label_exists:
        # Get the column name of the last column
        last_col_name = df.columns[-1]
        # Renaming the last column as "label"
        df.rename(columns={last_col_name: "label"}, inplace=True)

    if nrows is not None and isinstance(nrows, int) and df.shape[0] >= nrows:
        df = df.iloc[:nrows, :]
    print("df----", df)
    return df

#  python3 create_windows_datasetv2.py  --window 60

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		prog='Create temporary/experiment-specific dataset',
		description='This function creates a dataset of the size you want.  The data that will be used are set into the config file',
		epilog='Be careful where you save the generated dataset'
	)

	parser.add_argument('-n', '--name', type=str, help='name of save the dataset', default="Forecasting")
	parser.add_argument('-s', '--save_dir', type=str, help='path to save the dataset', default='dataset/')
	parser.add_argument('-p', '--path', type=str, help='path of the dataset to divide', default="dataset/forecasting/")
	parser.add_argument('-mp', '--metric_path', type=str, help='path to the metrics of the dataset given', default=FXX_metrics_path)
	# parser.add_argument('-w', '--window_size', type=str, help='window size to segment the timeseries to', required=True)
	parser.add_argument('-m', '--metric', type=str, help='metric to use to produce the labels', default='mae')
	parser.add_argument('-ho', '--horizon', type=int, help='horizon length', default=96)
	parser.add_argument('-l', '--lookback', type=int, help='lookback length', default=336)
	parser.add_argument('-win', '--window', type=int, help='window size / stride length', default=60)
	args = parser.parse_args()

	seq_len = args.lookback+args.horizon
	save_dir =  os.path.join(args.save_dir, f"Forecasting_{seq_len}_{args.window}")
	metricpath = os.path.join(args.metric_path, 'horizon_'+str(args.horizon), 'lookback_'+str(args.lookback))
	create_tmp_dataset(
		name=args.name,
		save_dir=save_dir,
		data_path=args.path,
		metric_path=metricpath,
		window_size=seq_len,
		metric=args.metric,
		window=args.window
	)


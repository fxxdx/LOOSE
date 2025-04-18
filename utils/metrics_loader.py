import os, glob

import numpy as np
import pandas as pd

from collections import Counter
from pathlib import Path


class MetricsLoader:
	"""Class to read, load and write metrics. The 'metrics' are the 
	evaluation metric results of the scores of the anomaly detectors 
	on the benchmark.
	"""
	def __init__(self, metrics_path):
		self.metrics_path = metrics_path

	def get_names(self):
		'''Return the names of all metrics in metrics path

		:return: list of names (strings) of metrics
		'''
		result = []
		n_detectors = len(os.listdir(self.metrics_path))
		# print("metrics_path:  ", self.metrics_path)
		for detector in os.listdir(self.metrics_path):
			# print("detector:  ", detector)
			for fname in glob.glob(os.path.join(self.metrics_path, detector, '*.csv')):
				# print("fname:  ", fname)
				result.append(fname)

		result = [name.split('/')[-1].replace('.csv', '') for name in result]
		print("result: ", result)
		result = Counter(result)
		# print("Counter result: ", result)
		# for elem in result:
		# 	if result[elem] != n_detectors:
		# 		raise ValueError('metrics occurances do not match for all detectors {}'.format(result))
		# print("list(result.keys() : ", list(result.keys()))
		return list(result.keys())

	
	def read(self, metric):
		'''Read the metrics and check that they all contain the same 
		timeseries and in the same order

		:param metric: name of metric that you want to load
		:return: dataframe of metrics' values
		'''
		df = []

		# Check if metric exists
		if metric not in self.get_names():
			raise ValueError(f"{metric} metric is not one of existing metrics")
		
		for detector in os.listdir(self.metrics_path):
			for fname in glob.glob(os.path.join(self.metrics_path, detector, metric + '.csv')):
				curr_df = pd.read_csv(fname, index_col = 0)
				# print("curr_df: ", curr_df)
				df.append(curr_df.sort_index())
				# print("curr_df.sort: ", curr_df.sort_index())
				# Check for consistency (can be disabled)
				if len(df) > 1 and not np.all(df[-1].index == df[-2].index):
					raise ValueError('timeseries in metric files do not match, {} != {}'.
						format(df[-1].shape, df[-2].shape))

		return pd.concat(df, axis=1)

	def write(self, data, files_names, detector, metric):
		'''Write a new metric

		:param detector: name of detector (string)
		:param metric: name of metric (string)
		'''

		# Create detector's directory
		Path(os.path.join(self.metrics_path, detector)).mkdir(parents=True, exist_ok=True)

		df = pd.DataFrame(data=data, index=files_names, columns=[detector])
		df.to_csv(os.path.join(self.metrics_path, detector, metric + '.csv'))
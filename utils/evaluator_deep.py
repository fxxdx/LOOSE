########################################################################
#
# @author : Emmanouil Sylligardos
# @when : Winter Semester 2022/2023
# @where : LIPADE internship Paris
# @title : MSAD (Model Selection Anomaly Detection)
# @component: utils
# @file : evaluator
#
########################################################################

import os
import pickle
from pathlib import Path
from collections import Counter
from time import perf_counter
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

from momentfm import MOMENTPipeline
from utils.multiseries_dataset import TimeseriesDataset, create_splits
from utils.config import *

import torch
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd


class Evaluator:
	"""A class with evaluation tools
	"""

	def predict(
		self,
		model,
		data_path,
		sequence,
		batch_size=16,
		window = 2,
		deep_model=True,
		device='cuda'
	):
		"""Predict function for all the models

		:param model: the object model whose predictions we want
		:param fnames: the names of the timeseries to be predicted
		:param data_path: the path to the timeseries 
			(please check that path and fnames together make the complete path)
		:param batch_size: the batch size used to make the predictions
		:param deep_model:
		:return df: dataframe with timeseries and predictions per time series
		"""

		# Setup
		all_preds = []
		inf_time = []

		# loop = tqdm(
		# 	fnames,
		# 	total=len(fnames),
		# 	desc="Computing",
		# 	unit="files",
		# 	leave=True
		# )

		# Main loop
		fnames = []
		for name in dataset_names:
		# for name in ['Weather.csv','AQWan.csv']:
			data_set = []

			# PDdata = pd.read_csv(os.path.join(data_path, name), index_col=0)
			PDdata = pd.read_csv(os.path.join(data_path, name))
			print("------------data----------")
			print(PDdata)
			data_idx, _ = create_splits(PDdata, sequence, split_per=1.0)
			# Replace indexes with file names
			# data_set.extend(PDdata.iloc[x*window:x*window + sequence, : ].to_numpy() for x in data_idx)
			data_set.extend(PDdata.iloc[x*sequence:(x+1)*sequence,:].to_numpy() for x in data_idx)
			fname_idx = f"{name}"

			# Fetch data for this specific timeseries
			data = TimeseriesDataset(
				data_path=data_path,
				dataset=data_set,
				verbose=False
			)

			if deep_model:
				tic = perf_counter()
				preds, timess= self.predict_timeseries(model, data, batch_size=batch_size, device=device)
			else:
				X_val, y_val = data.__getallsamples__().astype('float32'), data.__getalllabels__()
				tic = perf_counter()
				preds, timess = self.predict_timeseries_non_deep(model, X_val, y_val)

			# Compute metric value

			counter = Counter(preds)
			print("counter: ", counter)
			most_voted = counter.most_common(1)
			# print("fname_idx", fname_idx)
			# print("len(fname_idx)", len(fname_idx))
			# print("most voted: ", list(most_voted))
			toc = perf_counter()
			
			# Save info
			# Save info
			print(name)
			print("time: ", timess)
			print("choose: ", most_voted[0][0], forcast_names[most_voted[0][0]])
			all_preds.append(forcast_names[most_voted[0][0]])
			inf_time.append(timess)
			# all_preds.append([forcast_names[i] for i in preds])
			# print("all preds: ", all_preds)
			# print("len(all_preds): ", len(all_preds))
			# print("len(all_preds[0]): ", len(all_preds[0]))
			# inf_time.append(toc-tic)
			fnames.append(fname_idx)


		return pd.DataFrame(data=zip(all_preds, inf_time), columns=["class", "inf"], index=fnames)


	def predict_timeseries(self, model, val_data, batch_size, device='cuda', k=1):
		all_preds = []
		all_time = []
		# Timeseries to batches
		val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)


		for (inputs, mask, labels) in val_loader:
			tic = perf_counter()
			# Move data to the same device as model
			inputs = inputs.to(device)
			labels = labels.to(device)

			# Make predictions
			outputs = model(inputs.float())

			# Compute topk acc
			preds = outputs.argmax(dim=1)
			toc = perf_counter()
			all_time.append(toc - tic)
			all_preds.extend(preds.tolist())

		return all_preds, np.average(all_time)


	def predict_timeseries_non_deep(self, model, X_val, y_val):
		all_preds = []
		all_acc = []
		
		# Make predictions
		preds = model.predict(X_val)

		# preds = outputs.argmax(dim=1)
		# acc = (y_val == preds).sum() / y_val.shape[0]

		# all_acc.append(acc)
		all_preds.extend(preds.tolist())

		return all_preds


def save_classifier(model, path, fname=None):
	# Set up
	timestamp = datetime.now().strftime('%d%m%Y_%H%M%S')
	fname = f"model_{timestamp}" if fname is None else fname

	# Create saving dir if we need to
	filename = Path(os.path.join(path, f"{fname}.pkl"))
	filename.parent.mkdir(parents=True, exist_ok=True)

	# Save
	with open(filename, 'wb') as output:
		pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)

	return str(filename)


def load_classifier(path):
	"""Loads a classifier/model that is a pickle (.pkl) object.
	If the path is only the path to the directory of a given class
	of models, then the youngest model of that class is retrieved.

	:param path: path to the specific classifier to load,
		or path to a class of classifiers (e.g. rocket)
	:return output: the loaded classifier
	"""

	# If model is not given, load the latest
	if os.path.isdir(path):
		models = [x for x in os.listdir(path) if '.pkl' in x]
		models.sort(key=lambda date: datetime.strptime(date, 'model_%d%m%Y_%H%M%S.pkl'))
		path = os.path.join(path, models[-1])
	elif '.pkl' not in path:
		raise ValueError(f"Can't load this type of file {path}. Only '.pkl' files please")

	filename = Path(path)
	with open(f'{filename}', 'rb') as input:
		output = pickle.load(input)
	
	return output

'''
	def predict_non_deep(self, model, X_val, y_val):
		all_preds = []
		all_acc = []
		
		# Make predictions
		preds = model.predict(X_val)

		# preds = outputs.argmax(dim=1)
		# acc = (y_val == preds).sum() / y_val.shape[0]

		# all_acc.append(acc)
		all_preds.extend(preds.tolist())

		return all_preds
'''
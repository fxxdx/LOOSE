import os
from tqdm import tqdm
from utils.config import *
import re
import pandas as pd
import numpy as np
import math
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight

import torch
from torch.utils.data import Dataset



def create_splits(data, sequence, split_per=0.7, seed=None):
	# Read splits from file if provided
	length = data.shape[0]
	# sample_idx = math.floor((length-sequence)/window) + 1
	sample_idx = length / sequence
	# print("----------", sample_idx)
	# train_sample = math.ceil((length / sequence) * split_per)
	# train_sample = math.ceil((length-sequence+1) * split_per)
	train_sample = math.floor(sample_idx * split_per)
	# val_sample = (length / sequence) - train_sample
	val_sample = sample_idx-train_sample

	# Select random files for each subset
	train_idx = np.random.choice(
		# np.arange(int(length / sequence)),
		np.arange(int(sample_idx)),
		size=train_sample,
		replace=False
	)
	# val_idx = np.asarray([x for x in range(int(length / sequence)) if x not in train_idx])
	val_idx = np.asarray([x for x in range(int(sample_idx)) if x not in train_idx])
	return train_idx, val_idx


def read_files(data_path):
	"""Returns a list of names of the csv files in the 
	directory given.

	:param data_path: path to the directory/-ies with csv time series files
	:return fnames: list of strings
	"""

	# Load everything you can find
	fnames = [x for x in os.listdir(data_path) if ".csv" in x and "tsfresh" not in x.lower()]
	
	if len(fnames) > 0:
		pass
		# dataset = data_path.split('/')[-1]
		# dataset = dataset if len(dataset) > 0 else data_path.split('/')[-2]
		# fnames = [os.path.join(dataset, x) for x in fnames]
	else:
		datasets = [x for x in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, x))]
		for dataset in datasets:
			
			# Read file names
			curr_fnames = os.listdir(os.path.join(data_path, dataset))
			curr_fnames = [os.path.join(dataset, x) for x in curr_fnames]
			fnames.extend(curr_fnames)

	return fnames



class TimeseriesDataset(Dataset):
	def __init__(self, data_path, dataset, verbose=True):
		self.data_path = data_path
		self.data = dataset
		self.labels = []
		self.samples = []
		self.seq_len = int(re.search(r'\d+', str(data_path)).group())
		if len(self.data) == 0:
			return

		# Read datasets
		print("11111111111111111111111111111")
		# self.data = np.delete(self.data, 0, axis=-1)
		for row in self.data:
			self.labels.append(row[0][-1])
			self.samples.append(np.delete(row, -1, axis=1))
		print("2222222222222222222222222222222")
		# Concatenate samples and labels
		self.labels = np.asarray(self.labels)
		# self.samples = np.array(self.samples)
		# self.samples = np.concatenate(self.samples, axis=1)
		# # Add channels dimension
		# self.samples = self.samples[:, np.newaxis, :]
		
	def __len__(self):
		return self.labels.size

	def __getitem__(self, idx):

		timeseries_len = self.samples[idx].shape[1]
		seq_len = self.samples[idx].shape[0]
		input_mask = np.ones((seq_len, timeseries_len))
		# input_mask[self.seq_len - timeseries_len] = 0
		return self.samples[idx], input_mask, self.labels[idx]

	def __getallsamples__(self):
		return self.samples

	def __getalllabels__(self):
		return self.labels

	# def getallindex(self):
	# 	return self.indexes

	def __getlabel__(self, idx):
		return self.labels[idx]

	def my_collate(batch):
		data = [item[0] for item in batch]
		mask = [item[1] for item in batch]
		target = [item[2] for item in batch]
		return [data, mask, target]

	def get_weights_subset(self, device):
		'''Compute and return the class weights for the dataset'''

		# Count labels within those indices
		labels = np.fromiter(map(self.__getlabel__, range(self.__len__())), dtype=np.int16)

		# Compute weights
		labels_exist = np.unique(labels)
		labels_not_exist = [x for x in np.arange(len(detector_names)) if x not in labels_exist]
		sklearn_class_weights = list(compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels))
		for i in labels_not_exist:
			sklearn_class_weights.insert(i, 1)

		# Test
		# print('------------------------------------------')
		# counter = Counter(labels)
		# for detector, weight in zip(detector_names, sklearn_class_weights):
		# 	print(f'{detector} : {counter[detector_names.index(detector)]}, {weight:.3f}')

		return torch.Tensor(sklearn_class_weights).to(device)

import os
from tqdm import tqdm
import re
import pandas as pd
import numpy as np
import math
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight

import torch
from torch.utils.data import Dataset



def create_splits(data_path, split_per=0.7, seed=None, read_from_file=None):
	train_set = []
	val_set = []
	test_set = []
	# dir_path = os.path.split(data_path)[0]
	dir_path = data_path
	
	# Set seed if provided
	if seed: 
		np.random.seed(seed)

	# Read splits from file if provided
	if read_from_file is not None:
		df = pd.read_csv(read_from_file, index_col=0)
		subsets = list(df.index)
		print("-----------------")
		print(subsets)
		print("-----------------")
		if 'train_set' in subsets and 'val_set' in subsets:
			train_set = [x for x in df.loc['train_set'].tolist() if not isinstance(x, float) or not math.isnan(x)]
			val_set = [x for x in df.loc['val_set'].tolist() if not isinstance(x, float) or not math.isnan(x)]

			return train_set, val_set, test_set
		elif 'train_set' in subsets and 'test_set' in subsets:
			train_val_set = [x for x in df.loc['train_set'].tolist() if not isinstance(x, float) or not math.isnan(x)]
			test_set = [x for x in df.loc['test_set'].tolist() if not isinstance(x, float) or not math.isnan(x)]

			datasets = list(set([x.split('/')[0] for x in train_val_set]))
			datasets.sort()
		else:
			raise ValueError('Did not expect this type of file.')
	else:
		datasets = [x for x in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, x))]
		datasets.sort()

	if not os.path.isdir(dir_path):
		dir_path = '/'.join(dir_path.split('/')[:-1])
	
	# Random split of train & val sets
	for dataset in datasets:
		# Read file names
		fnames = os.listdir(os.path.join(dir_path, dataset))

		# Decide on the size of each subset
		n_timeseries = len(fnames)
		train_split = math.ceil(n_timeseries * split_per)
		val_split = n_timeseries - train_split

		# Select random files for each subset
		train_idx = np.random.choice(
			np.arange(n_timeseries), 
			size=train_split, 
			replace=False
		)
		val_idx = np.asarray([x for x in range(n_timeseries) if x not in train_idx])

		# Replace indexes with file names
		train_set.extend([os.path.join(dataset, fnames[x]) for x in train_idx])
		val_set.extend([os.path.join(dataset, fnames[x]) for x in val_idx])
	
	return train_set, val_set, test_set


def read_files(data_path):

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
	def __init__(self, data_path, fnames, verbose=True):
		self.data_path = data_path
		self.fnames = fnames
		self.labels = []
		self.samples = []
		self.indexes = []
		self.seq_len = int(re.search(r'\d+', str(data_path)).group())
		if len(self.fnames) == 0:
			return
		print("seq_len: ", self.seq_len)
		# Read datasets
		for fname in tqdm(self.fnames, disable=not verbose, desc="Loading dataset"):
			# print("fname: ", fname)
			# print("datapath: ",self.data_path)
			data = pd.read_csv(os.path.join(self.data_path, fname), index_col=0)
			dataset = fname.split('/')[0]
			curr_idxs = list(data.index)
			curr_idxs = [os.path.join(dataset, x) for x in curr_idxs]

			self.indexes.extend(curr_idxs)	
			self.labels.extend(data['label'].tolist())
			self.samples.append(data.iloc[:, 1:].to_numpy())
		
		# Concatenate samples and labels
		self.labels = np.asarray(self.labels)
		self.samples = np.concatenate(self.samples, axis=0)

		# Add channels dimension
		self.samples = self.samples[:, np.newaxis, :]
		
	def __len__(self):
		return self.labels.size

	def __getitem__(self, idx):
		timeseries_len = len(self.samples[idx])
		input_mask = np.ones(self.seq_len)
		input_mask[self.seq_len - timeseries_len] = 0
		return self.samples[idx], input_mask, self.labels[idx]

	def __getallsamples__(self):
		return self.samples

	def __getalllabels__(self):
		return self.labels

	def getallindex(self):
		return self.indexes

	def __getlabel__(self, idx):
		return self.labels[idx]

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

from utils.metrics_loader import MetricsLoader
from utils.config import FXX_metrics_path, FXX_data_path, forcast_names, FXX_acc_tables_path

import os
import pandas as pd
import argparse
import numpy as np
from natsort import natsorted
import pathlib
from pathlib import Path
'''
将各个预测方法跑各个数据集得到的结果合成一个文件，取名MergedTable_{metric}.csv

'''
def merge_scores(metric, save_path, metric_path, horizon, lookback):
	# Load MetricsLoader object
	metricsloader = MetricsLoader(metric_path)
	
	# Check if given metric exists
	# if metric.upper()  not in metricsloader.get_names():
	if metric not in metricsloader.get_names():
		raise ValueError(f"Not recognizable metric {metric}. Please use one of {metricsloader.get_names()}")

	total_pd = []
	for method in os.listdir(metric_path):
		if method in forcast_names:
			file_path = os.path.join(metric_path, method, str(metric)+'.csv')
			current_pd = pd.read_csv(file_path,index_col=0)
			print(current_pd.shape)
			total_pd.append(current_pd)

	column_name = ['dataset'] + forcast_names
	# print(pd.DataFrame(data=total_pd, columns=column_name))
	t = pd.concat([i for i in total_pd], axis=1)
	target_path = Path(os.path.join(save_path, f'lookback_'+str(lookback), f'horizon_'+ str(horizon)))
	pathlib.Path(target_path).mkdir(parents=True, exist_ok=True)
	t.to_csv(os.path.join(target_path,f'MergedTable_{metric}.csv'))


def merge_inference_times(path, save_path):
	detectors_inf_time_path = 'results/execution_time/detectors_inference_time.csv'

	# Read raw predictions of each model selector and fix indexing
	selector_predictions = {}
	for file_name in os.listdir(path):
		if file_name.endswith('.csv'):
			selector_name = file_name.replace("_preds.csv", "")
			curr_df = pd.read_csv(os.path.join(path, file_name), index_col=0)
			old_indexes = curr_df.index.tolist()
			old_indexes_split = [tuple(x.split('/')) for x in old_indexes]
			filenames_df = pd.DataFrame(old_indexes_split, index=old_indexes, columns=['dataset', 'filename'])
			curr_df = pd.merge(curr_df, filenames_df, left_index=True, right_index=True)
			curr_df = curr_df.set_index(keys=['dataset', 'filename'])
			selector_predictions[selector_name] = curr_df

	# Read detectors inference time
	try:
		detectors_inf = pd.read_csv(detectors_inf_time_path, index_col=["dataset", "filename"])
	except Exception as e :
		print("Oops could read detectors' inference times file. Please specify the path correctly in code")
		print(e)
		return

	# For each time series read the predicted detector and add the inference time of that detector to its prediction time
	final_df = None
	for selector_name, df in selector_predictions.items():
		df = df[df.index.isin(detectors_inf.index)]

		sum = np.diag(detectors_inf.loc[df.index, df[f"{selector_name}_class"]]) + df[f"{selector_name}_inf"]
		sum.rename(f"{selector_name}", inplace=True)

		sum_df = pd.merge(df, sum, left_index=True, right_index=True)
		sum_df.rename(columns={f"{selector_name}_inf": f"{selector_name}_pred"}, inplace=True)
		
		if final_df is None:
			final_df = sum_df
		else:
			final_df = pd.merge(final_df, sum_df, left_index=True, right_index=True)


	# Merge with detectors inference times
	final_df = pd.merge(detectors_inf, final_df, left_index=True, right_index=True)
	
	# Save the file with name model_selectors_inference_time.csv in the results/execution_time dir
	final_df.to_csv(os.path.join(save_path, f'current_inference_time.csv'), index=True)
	print(final_df)



if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		prog='Merge scores',
		description="Merge all models' scores into one csv"
	)
	parser.add_argument('-m', '--metric', help='metric to use', type =str, default='mae')
	parser.add_argument('-s', '--save_path', help='where to save the result', default = 'dataset/acc_tables/')
	parser.add_argument('-time', '--time-true', action="store_true", help='whether to produce time results')
	parser.add_argument('-metric', '--metric_path', help='metric path', type =str, default='dataset/metric/')
	parser.add_argument('-ho', '--horizon', type=int, help='batch size', default=60)
	parser.add_argument('-lo', '--lookback', type=int, help='batch size', default=336)
	args = parser.parse_args()

	metric_path = os.path.join(args.metric_path, f'horizon_' + str(args.horizon), f'lookback_' + str(args.lookback))
	if not args.time_true:
		merge_scores(
			metric=args.metric,
			save_path=args.save_path,
			metric_path = metric_path,
			horizon = args.horizon,
			lookback = args.lookback
		)
	else:
		merge_inference_times(
			save_path=args.save_path,
			metric_path=metric_path,
		)


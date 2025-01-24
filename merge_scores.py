from utils.metrics_loader import MetricsLoader
from utils.config import FXX_metrics_path, FXX_data_path, forcast_names, FXX_acc_tables_path

import os
import pandas as pd
import argparse
import numpy as np
from natsort import natsorted
from pathlib import Path

def merge_scores(path, metric, save_path, metric_path, horizon, lookback, window_size, model_name):
	# Load MetricsLoader object
	metricsloader = MetricsLoader(metric_path)
	
	# Check if given metric exists
	# if metric.upper()  not in metricsloader.get_names():
	if metric not in metricsloader.get_names():
		raise ValueError(f"Not recognizable metric {metric}. Please use one of {metricsloader.get_names()}")

	# Read accuracy table, fix filenames & indexing, remove detectors scores
	acc_tables_path = os.path.join(FXX_acc_tables_path, f'lookback_'+str(lookback), f'horizon_'+ str(horizon), "MergedTable_{}.csv".format(metric))
	acc_tables = pd.read_csv(acc_tables_path)
	acc_tables.columns.values[0] = 'filename'

	acc_tables['filename'] = acc_tables['filename'].apply(lambda x: x.replace('.txt', '.out') if x.endswith('.txt') else x)
	acc_tables.set_index(['filename'], inplace=True)
	FName = forcast_names
	acc_tables.drop(columns=FName, inplace=True)

	# Read detectors and oracles scores
	metric_scores = metricsloader.read(metric)
	
	# Read classifiers predictions, and add scores
	df = None
	# scores_files = [x for x in os.listdir(path) if '.csv' in x and str(horizon) in x]
	scores_files = model_name+'_'+str(horizon+lookback)+'_'+str(horizon)+'_'+str(window_size)+'_preds.csv'
	file_path = os.path.join(path, scores_files)
	current_classifier = pd.read_csv(file_path, index_col=0)

	col_name = [x for x in current_classifier.columns if "class" in x][0]

	values = np.diag(metric_scores.loc[current_classifier.index, current_classifier.iloc[:, 0]])
	curr_df = pd.DataFrame(values, index=current_classifier.index, columns=[col_name.replace("_class", "")])
	curr_df = pd.merge(current_classifier[col_name], curr_df, left_index=True, right_index=True)

	if df is None:
		df = curr_df
	else:
		df = pd.merge(df, curr_df, left_index=True, right_index=True)
	df = df.reindex(natsorted(df.columns, key=lambda y: y.lower()), axis=1)

	df = pd.merge(df, metric_scores[forcast_names], left_index=True, right_index=True)

	# Add true labels from MAE metrics
	auc_pr_detectors_scores = metricsloader.read('mae')[forcast_names]
	labels = auc_pr_detectors_scores.idxmin(axis=1).to_frame(name='label')
	df = pd.merge(labels, df, left_index=True, right_index=True)
	
	# Change the indexes to dataset, filename
	old_indexes = df.index.tolist()
	old_indexes_split = [tuple(x.split('/')) for x in old_indexes]
	filenames_df = pd.DataFrame(old_indexes_split, index=old_indexes, columns=['filename'])
	df = pd.merge(df, filenames_df, left_index=True, right_index=True)
	df = df.set_index(keys=['filename'])

	# Merge the two dataframes now that they have common indexes
	final_df = df.join(acc_tables)
	indexes_not_found = final_df[final_df.iloc[:, -len(acc_tables.columns):].isna().any(axis=1)].index.tolist()


	# Save the final dataframe
	Path(os.path.join(save_path, model_name)).mkdir(parents=True, exist_ok=True)
	final_df.to_csv(os.path.join(save_path, model_name, f'current_accuracy_{horizon}_{lookback}_{window_size}_{metric}.csv'), index=True)


def merge_inference_times(path, save_path):
	detectors_inf_time_path = 'results/execution_time/detectors_inference_time.csv'

	# Read raw predictions of each model selector and fix indexing
	selector_predictions = {}
	for file_name in os.listdir(path):
		if file_name.endswith('.csv'):
			selector_name = file_name.replace("_preds.csv", "")
			curr_df = pd.read_csv(os.path.join(path, file_name), index_col=0)
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


'''
在dataset/acc_tables/里存的文件的基础之上  添加 path路径中（results/raw_predictions/）存储的模型跑的结果
'''
if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		prog='Merge scores',
		description="Merge all models' scores into one csv"
	)
	parser.add_argument('-p', '--path', help='path of the files to merge', default='results/raw_predictions/')
	parser.add_argument('-m', '--metric', help='metric to use', type =str, default='mae')
	parser.add_argument('-s', '--save_path', help='where to save the result', default = 'results/mergescore/')
	parser.add_argument('-time', '--time-true', action="store_true", help='whether to produce time results')
	parser.add_argument('-metric', '--metric_path', help='metric path', type =str, default='dataset/metric/')
	parser.add_argument('-ho', '--horizon', type=int, help='batch size', default=60)
	parser.add_argument('-lo', '--lookback', type=int, help='batch size', default=336)
	parser.add_argument('-win', '--window_size', type=int, help='window_size', default=100)
	parser.add_argument('-n', '--model_name', type=str, help='model name', default='moment')

	args = parser.parse_args()

	metric_path  = os.path.join(args.metric_path, f'horizon_'+str(args.horizon), f'lookback_'+str(args.lookback))
	if not args.time_true:
		merge_scores(
			path=args.path, 
			metric=args.metric,
			save_path=args.save_path,
			metric_path = metric_path,
			horizon=args.horizon,
			lookback=args.lookback,
			window_size = args.window_size,
			model_name= args.model_name
		)
	else:
		merge_inference_times(
			path=args.path, 
			save_path=args.save_path
		)


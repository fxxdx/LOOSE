import argparse
from collections import Counter
from pathlib import Path
import torch

from momentfm import MOMENTPipeline
from utils.config import *
from utils.evaluator import Evaluator


def eval_deep_model(
	data_path, 
	model_name, 
	model_path,
	horizon,
	lookback,
	batch_size,
	window,
	model_parameters_file=None, 
	path_save=None,
	read_from_file=None,
	model=None,
):
	"""Evaluate a deep learning model on time series data and predict the time series.

    :param data_path: Path to the time series data.
    :param model_name: Name of the model to be evaluated.
    :param model_path: Path to the pretrained model weights.
    :param horizon: the length of the forecasting time-series data
    :param lookback: the length of the historial data
    :param model_parameters_file: Path to the JSON file containing model parameters.
    :param path_save: Path to save the evaluation results.
    :param read_from_file: Path to the file containing split information.
    :param model: Preloaded model instance.

	Returns:
    DataFrame: A DataFrame containing the predicted time series.
	"""
	window_size = data_path.split('_')[-1]
	window_size = window_size.replace('/', '')
	print("-------------window_size-------------", window_size)
	sequence = horizon + lookback
	Path(os.path.join(path_save)).mkdir(parents=True, exist_ok=True)

	assert(
		(model is not None) or \
		(model_path is not None and model_parameters_file is not None)
	), "You should provide the model or the path to the model, not both"


	# Load the model only if not provided
	if model == None:

		
		# Load model
		classifier_name = f"{model_name}_{sequence}"
		model_path = os.path.join(model_path, classifier_name, f"horizon_{horizon}")

		model = MOMENTPipeline.from_pretrained(
			"AutonLab/MOMENT-1-large",
			model_kwargs={"task_name": "classification",
						  "n_channels": sequence,
						  "num_class": 16,
						  'reduction': 'concat'}
		)
		model.init()
		# Check if model_path is specific file or dir
		if os.path.isdir(model_path):
			# Check the number of files in the directory
			files = os.listdir(model_path)
			if len(files) == 1:
				# Load the single file from the directory
				model_path = os.path.join(model_path, files[0])
			else:
				raise ValueError("Multiple files found in the 'model_path' directory. Please provide a single file or specify the file directly.")

		if torch.cuda.is_available():
			model.load_state_dict(torch.load(model_path))
			model.eval()
			model.to('cuda')
		else:
			model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
			model.eval()
		print("complete!!!!")


	if read_from_file is not None and "unsupervised" in read_from_file:
		classifier_name += f"_{read_from_file.split('/')[-1].replace('unsupervised_', '')[:-len('.csv')]}"
	elif "unsupervised" in path_save:
		extra = model_path.split('/')[-2].replace(classifier_name, "")
		classifier_name += extra
	
	# Evaluate model
	evaluator = Evaluator()
	results = evaluator.predict(
		model=model,
		data_path=data_path,
		sequence= horizon + lookback,
		batch_size=batch_size,
		window = window,
		deep_model= True,
		device='cuda' if torch.cuda.is_available() else 'cpu',
	)
	results = results.sort_index()
	results.columns = [f"{classifier_name}_{x}" for x in results.columns.values]
	
	# Print results
	print("results: ", results)
	counter = dict(Counter(results[f"{classifier_name}_class"]))
	counter = {k: v for k, v in sorted(counter.items(), key=lambda item: item[1], reverse=True)}
	print("counter: ", counter)
	
	# Save the results
	if path_save is not None:
		file_name = os.path.join(path_save, f"{classifier_name}_{horizon}_{window_size}_preds.csv")
		results.to_csv(file_name)

	return results

'''
利用create window datasetv2 创建好的固定序列长度的数据集（dataset/Forecasting_396/）评估TrainClaasiffier训练的“最佳方法选择模型”
'''
if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		prog='Evaluate deep learning models',
		description='Evaluate all deep learning architectures on a single or multiple time series \
			and save the results'
	)
	
	parser.add_argument('-d', '--data', type=str, help='path to the time series to predict',default = 'dataset/Forecasting_396_100/')
	parser.add_argument('-m', '--model', type=str, help='model to run', default='loose')
	parser.add_argument('-mp', '--model_path', type=str, help='path to the trained model', default = 'results/weights/')
	parser.add_argument('-pa', '--params', type=str, help="a json file with the model's parameters", default='results/weights/')
	parser.add_argument('-ps', '--path_save', type=str, help='path to save the results', default="results/raw_predictions")
	parser.add_argument('-f', '--file', type=str, help='path to file that contains a specific split', default=None)
	parser.add_argument('-ho', '--horizon', type=int, help='batch size', default=60)
	parser.add_argument('-lo', '--lookback', type=int, help='batch size', default=336)
	parser.add_argument('-b', '--batch', type=int, help='batch size', default=1)
	parser.add_argument('-win', '--window', type=int, help='window size', default=2)

	args = parser.parse_args()
	eval_deep_model(
		data_path=args.data, 
		model_name=args.model, 
		model_path=args.model_path,
		horizon=args.horizon,
		lookback=args.lookback,
		batch_size = args.batch,
		window = args.window,
		model_parameters_file=args.params,
		path_save=args.path_save,
		read_from_file=args.file
	)
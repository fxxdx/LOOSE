from models.model.convnet import ConvNet
from models.model.inception_time import InceptionModel
from models.model.resnet import ResNetBaseline
from models.model.sit import SignalTransformer
import os
# Important paths
FXX_data_path = "dataset/forecasting/"
FXX_metrics_path = "dataset/metric/"
FXX_scores_path = "dataset/scores/"
FXX_acc_tables_path = "dataset/acc_tables/"

save_done_training = 'results/done_training/'	# when a model is done training a csv with training info is saved here
path_save_results = 'results/raw_predictions'	# when evaluating a model, the predictions will be saved here

# Detector

forcast_names = [
    'Crossformer',
    'DLinear',
    'FEDformer',
    'FiLM',
    'Informer',
    'Linear',
    'MICN',
    'NLinear',
    'Nonstationary_Transformer',
    'PatchTST',
    'regressionmodel',
    'RNN',
    'TCN',
    'TimesNet',
    'Triformer',
    'VAR'
]


result = os.listdir('scripts/multivariate_forecast/horizon_60')
dataset_names = [(name.split('_')[1]).split('.')[0]+'.csv' for name in result]

dataset_names_nonecsv = [name.split('_')[0] for name in result]

deep_models = {
	'convnet':ConvNet,
	# 'inception_time':InceptionModel,
	'inception':InceptionModel,
	'resnet':ResNetBaseline,
	'sit':SignalTransformer,
}

# for i in range(0,len(dataset_names)):
#     print(i, dataset_names[i])
#     if dataset_names[i] =='other.csv': del dataset_names[i]
# Dict of model names to Constructors
# deep_models = {
# 	'convnet':ConvNet,
# 	'inception_time':InceptionModel,
# 	'inception':InceptionModel,
# 	'resnet':ResNetBaseline,
# 	'sit':SignalTransformer,
# }

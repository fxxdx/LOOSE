import os
import argparse
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from pprint import pprint
import torch
from momentfm.data.anomaly_detection_dataset import AnomalyDetectionDataset
from torch.utils.data import DataLoader
from momentfm.utils.masking import Masking
from tqdm import *
import re
import numpy as np
from datetime import datetime
import pandas as pd
import math
from momentfm.utils.anomaly_detection_metrics import adjbestf1
from momentfm import MOMENTPipeline
from utils.config import *
from utils.multiseries_dataset import create_splits, TimeseriesDataset
from utils.train_model_utils import ModelExecutioner
import torch.nn as nn
from pathlib import Path


def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return [data, target]

def get_weights_subset(device):
    '''Compute and return the class weights for the dataset'''

    # Count labels within those indices
    labels = np.fromiter(map(self.__getlabel__, range(self.__len__())), dtype=np.int16)

    # Compute weights
    labels_exist = np.unique(labels)
    labels_not_exist = [x for x in np.arange(len(detector_names)) if x not in labels_exist]
    sklearn_class_weights = list(compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels))
    for i in labels_not_exist:
        sklearn_class_weights.insert(i, 1)


    return torch.Tensor(sklearn_class_weights).to(device)


def load_moment(n_channels):
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={"task_name": "classification",
            "n_channels": n_channels,
            "num_class": 16,
            'reduction': 'concat'
            },
    )
    model.init()
    # print(model)
    return model


def train_deep_model(
        data_path,
        model_name,
        batch_size,
        epochs,
        horizon,
        lookback,
):
    # Set up
    window_size = int(re.search(r'\d+', str(args.path)).group())
    device = 'cuda'
    save_runs = 'results/runs/'
    save_weights = 'results/weights/'
    sequence = horizon + lookback
    n_channels = sequence
    model = load_moment(n_channels).to(device)
    inf_time = True  # compute inference time per timeseries



    # Create the model, load it on GPU and print it
    classifier_name = f"{model_name}_{sequence}"
    # if read_from_file is not None and "unsupervised" in read_from_file:
    #     classifier_name += f"_{read_from_file.split('/')[-1].replace('unsupervised_', '')[:-len('.csv')]}"

    # Create the executioner object
    model_execute = ModelExecutioner(
        model=model,
        model_name=classifier_name,
        horizon=horizon,
        device=device,
        criterion=nn.CrossEntropyLoss().to(device),
        runs_dir=save_runs,
        weights_dir=save_weights,
        learning_rate=0.0001
    )

    # Check device of torch
    model_execute.torch_devices_info()
    train_set = []
    val_set = []
    for name in dataset_names:
        PDdata = pd.read_csv(os.path.join(data_path, name))
        # print("------------data----------")
        # print(PDdata)

        train_idx, val_idx = create_splits(PDdata, sequence)

        # Replace indexes with file names
        train_set.extend(PDdata.iloc[x*sequence:(x+1)*sequence,:].to_numpy() for x in train_idx)
        val_set.extend(PDdata.iloc[x*sequence:(x+1)*sequence,:].to_numpy() for x in val_idx)
        # train_set.extend(PDdata.iloc[x*window:x*window + sequence, :].to_numpy() for x in train_idx)
        # val_set.extend(PDdata.iloc[x*window:x*window + sequence, :].to_numpy() for x in val_idx)
    training_data = TimeseriesDataset(data_path, dataset=train_set)
    val_data = TimeseriesDataset(data_path, dataset=val_set)
        # print("------------datalenth----------")
        # print(len(training_data.__getallsamples__()))
        # print(len(val_data.__getallsamples__()))
        # Create the data loaders
    training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)


    # Run training procedure
    model, results = model_execute.train(
        n_epochs=epochs,
        training_loader=training_loader,
        validation_loader=validation_loader,
        verbose=True,
    )


    # Save training stats
    timestamp = datetime.now().strftime('%d%m%Y_%H%M%S')
    df = pd.DataFrame.from_dict(results, columns=["training_stats"], orient="index")

    Path(os.path.join(save_done_training, "lookback_"+str(lookback), "horizon_"+str(horizon))).mkdir(parents=True, exist_ok=True)


    df.to_csv(os.path.join(save_done_training, "lookback_"+str(lookback), "horizon_"+str(horizon),f"{classifier_name}_{timestamp}.csv"))


'''
利用create window datasetv2 创建好的固定序列长度的数据集（dataset/Forecasting_396/）训练“最佳方法选择模型”,训练结果会存放在'results/done_training/'  包含训练的时间
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='run_experiment',
        description='This function is made so that we can easily run configurable experiments'
    )

    parser.add_argument('-p', '--path', type=str, help='path to the dataset to use', default='dataset/Forecasting_432_60/')
    parser.add_argument('-s', '--split', type=float, default=0.7, help='split percentage for train and val sets')
    parser.add_argument('-se', '--seed', type=int, default=None, help='Seed for train/val split')
    parser.add_argument('-f', '--file', type=str, default=None, help='path to file that contains a specific split')
    parser.add_argument('-m', '--model', type=str, default='moment', help='model to run')
    parser.add_argument('-pa', '--params', type=str, default='models/configuration/convnet_default.json', help="a json file with the model's parameters")
    parser.add_argument('-b', '--batch', type=int, help='batch size', default=1)
    parser.add_argument('-ep', '--epochs', type=int, default = 20, help='number of epochs')
    parser.add_argument('-e', '--eval-true', action="store_true",
                        help='whether to evaluate the model on test data after training')
    parser.add_argument('-ho', '--horizon', type=int, help='batch size', default=60)
    parser.add_argument('-lo', '--lookback', type=int, help='batch size', default=336)


    args = parser.parse_args()
    train_deep_model(
        data_path=args.path,
        model_name=args.model,
        batch_size=args.batch,
        epochs=args.epochs,
        horizon = args.horizon,
        lookback = args.lookback
    )

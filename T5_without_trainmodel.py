import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import numpy as np
import random
import argparse
import torch
import torch.cuda.amp
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import re
from utils.informer_dataset import InformerDataset
from utils.forecasting_metrics import get_forecasting_metrics
import datetime
from chronos import BaseChronosPipeline

def control_randomness(seed: int = 13):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_moment(data_path, window, batch_size, horizon):
    print("5555555555551")
    model = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-t5-base",  # use "amazon/chronos-bolt-small" for the corresponding Chronos-Bolt model
    device_map="cuda",  # use "cpu" for CPU inference
    torch_dtype=torch.float16)

    # model.init()
    print("1111111111111111111")
    # Set random seeds for PyTorch, Numpy etc.
    control_randomness(seed=13)

    # Load the data
    print("222222222222222")
    train_dataset = InformerDataset(datapath=data_path, data_split="no", random_seed=13, forecast_horizon=horizon)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print("333333333333333333")
    criterion = torch.nn.MSELoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Move the model to the GPU
    # model = model.to(device)

    # Move the loss function to the GPU
    criterion = criterion.to(device)

    starttime = datetime.datetime.now()


    # Evaluate the model on the test split
    trues, preds, histories, losses, times = [], [], [], [], []
    # model.eval()
    with torch.no_grad():
        for timeseries, forecast, input_mask in tqdm(train_loader, total=len(train_loader)):
            # Move the data to the GPU
            # timeseries = timeseries.float().to(device)
            # forecast = forecast.float().to(device)
            start1 = datetime.datetime.now()

            # timeseries = timeseries.view(batch_size, -1)
            num_varia = timeseries.shape[-2]
            forecast = forecast.float()
            print(torch.squeeze(timeseries,0).shape)
            timeseries = timeseries.float()
            with torch.cuda.amp.autocast():
                # output = model(timeseries, input_mask)
                output_i, output = model.predict_quantiles(
                    context=torch.squeeze(timeseries,0),
                    prediction_length=horizon,
                    quantile_levels=[0.1, 0.5, 0.9],
                )
            currentime = datetime.datetime.now()

            print("final output: ", output.shape)

            output = torch.unsqueeze(output,0)
            print("final output: ", output.shape)

            # A, B, C = forecast.shape
            # forecast = forecast.view(B, C).transpose(1, 0)
            # print(forecast.shape)
            # output =torch.tensor(output).to(device)
            output = output.to(device)
            forecast = forecast.to(device)
            loss = criterion(output, forecast)
            losses.append(loss.item())

            times.append((currentime-start1).seconds)

            trues.append(forecast.detach().cpu().numpy())
            preds.append(output.detach().cpu().numpy())
            histories.append(timeseries.detach().cpu().numpy())

    losses = np.array(losses)
    average_loss = np.average(losses)

    times = np.array(times)
    average_time = np.average(times)
    trues = np.concatenate(trues, axis=0)
    preds = np.concatenate(preds, axis=0)
    histories = np.concatenate(histories, axis=0)

    metrics = get_forecasting_metrics(y=trues, y_hat=preds, reduction='mean')
    endtime = datetime.datetime.now()
    totaltime = (endtime - starttime).seconds
    print(f"File {data_path} | MSE: {metrics.mse:.4f} | MAE: {metrics.mae:.4f} | RMSE: {metrics.rmse:.4f} | Time: {totaltime:.3f} | Avg Time: {average_time:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='run_experiment',
        description='This function is made so that we can easily run configurable experiments'
    )

    parser.add_argument('-p', '--path', type=str, help='path to the dataset to use', default=None)
    parser.add_argument('-w', '--window', type=int, help='input_chunk_length', default=336)
    parser.add_argument('-b', '--batch_size', type=int, help='batch size', default=4)
    parser.add_argument('-ho', '--horizon', type=int, help='horizon', default=60)

    args = parser.parse_args()
    train_moment(
        data_path=args.path,
        window = args.window,
        batch_size= args.batch_size,
        horizon = args.horizon
    )

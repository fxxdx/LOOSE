from momentfm import MOMENTPipeline
import numpy as np
import argparse
import torch
import torch.cuda.amp
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import re
import datetime
from momentfm.utils.utils import control_randomness
from momentfm.data.informer_dataset import InformerDataset
from momentfm.utils.forecasting_metrics import get_forecasting_metrics

def train_moment(data_path, window, batch_size, horizon):
    # window_size = int(re.search(r'\d+', str(data_path)).group())
    window_size = window
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={
            'task_name': 'forecasting',
            'forecast_horizon': horizon,
            'head_dropout': 0.1,
            'weight_decay': 0,
            'freeze_encoder': True, # Freeze the patch embedding layer
            'freeze_embedder': True, # Freeze the transformer encoder
            'freeze_head': False, # The linear forecasting head must be trained
            'seq_len': window_size,
        },
    )
    model.init()

    # Set random seeds for PyTorch, Numpy etc.
    control_randomness(seed=13)

    # Load the data

    train_dataset = InformerDataset(datapath=data_path, data_split="no", random_seed=13, forecast_horizon=horizon)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Move the model to the GPU
    model = model.to(device)

    # Move the loss function to the GPU
    criterion = criterion.to(device)



    # Evaluate the model on the test split
    trues, preds, histories, losses, times = [], [], [], [], []
    model.eval()
    start_time = datetime.datetime.now()
    with torch.no_grad():
        for timeseries, forecast, input_mask in tqdm(train_loader, total=len(train_loader)):
            # Move the data to the GPU
            timeseries = timeseries.float().to(device)
            input_mask = input_mask.to(device)
            forecast = forecast.float().to(device)
            starttime = datetime.datetime.now()
            with torch.cuda.amp.autocast():
                output = model(timeseries, input_mask)
                currentime = datetime.datetime.now()

            loss = criterion(output.forecast, forecast)
            losses.append(loss.item())
            times.append((currentime-starttime).seconds)

            trues.append(forecast.detach().cpu().numpy())
            preds.append(output.forecast.detach().cpu().numpy())
            histories.append(timeseries.detach().cpu().numpy())
    endtime = datetime.datetime.now()
    totaltime = (endtime - start_time).seconds
    losses = np.array(losses)
    average_loss = np.average(losses)

    times = np.array(times)
    average_time = np.average(times)

    # preds = np.array(preds)
    # trues = np.array(trues)
    # preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    # trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    trues = np.concatenate(trues, axis=0)
    preds = np.concatenate(preds, axis=0)
    histories = np.concatenate(histories, axis=0)

    metrics = get_forecasting_metrics(y=trues, y_hat=preds, reduction='mean')
    print(f"File {data_path} | MSE: {metrics.mse:.3f} | MAE: {metrics.mae:.3f} | RMSE: {metrics.rmse:.4f} | Time: {totaltime:.3f} | Avg Time: {average_time:.8f}")

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

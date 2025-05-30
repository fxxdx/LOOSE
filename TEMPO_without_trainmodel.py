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
from tempo.models.TEMPO import TEMPO

def control_randomness(seed: int = 13):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_moment(data_path, window, batch_size, horizon):
    # window_size = int(re.search(r'\d+', str(data_path)).group())
    window_size = window
    # model = MOMENTPipeline.from_pretrained(
    #     "AutonLab/MOMENT-1-large",
    #     model_kwargs={
    #         'task_name': 'forecasting',
    #         'forecast_horizon': horizon,
    #         'head_dropout': 0.1,
    #         'weight_decay': 0,
    #         'freeze_encoder': True, # Freeze the patch embedding layer
    #         'freeze_embedder': True, # Freeze the transformer encoder
    #         'freeze_head': False, # The linear forecasting head must be trained
    #         'seq_len': window_size,
    #     },
    # )
    model = TEMPO.load_pretrained_model(
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        repo_id="Melady/TEMPO",
        filename="TEMPO-80M_v1.pth",
        cache_dir="./checkpoints/TEMPO_checkpoints"
    )
    # model.init()

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

    starttime = datetime.datetime.now()


    # Evaluate the model on the test split
    trues, preds, histories, losses, times = [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for timeseries, forecast, input_mask in tqdm(train_loader, total=len(train_loader)):
            # Move the data to the GPU
            timeseries = timeseries.float().to(device)
            input_mask = input_mask
            forecast = forecast.float().to(device)
            starttime1 = datetime.datetime.now()
            with torch.cuda.amp.autocast():
                # output = model(timeseries, input_mask)
                output = model.predict(timeseries, pred_length=horizon)
                currentime = datetime.datetime.now()
            # print(output.shape)
            A, B, C = forecast.shape
            forecast = forecast.view(B, C).transpose(1, 0)
            # print(forecast.shape)
            output =torch.tensor(output).to(device)
            loss = criterion(output, forecast)
            losses.append(loss.item())

            times.append((currentime-starttime1).seconds)

            trues.append(forecast.detach().cpu().numpy())
            preds.append(output.detach().cpu().numpy())
            histories.append(timeseries.detach().cpu().numpy())
    endtime = datetime.datetime.now()
    losses = np.array(losses)
    average_loss = np.average(losses)
    preds = np.array(preds)
    trues = np.array(trues)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

    times = np.array(times)
    average_time = np.average(times)
    # trues = np.concatenate(trues, axis=0)
    # preds = np.concatenate(preds, axis=0)

    histories = np.concatenate(histories, axis=0)

    metrics = get_forecasting_metrics(y=trues, y_hat=preds, reduction='mean')

    totaltime = (endtime - starttime).seconds
    print(f"File {data_path} | MSE: {metrics.mse:.4f} | MAE: {metrics.mae:.4f}  | RMSE: {metrics.rmse:.4f} | Time: {totaltime:.4f} | Avg Time: {average_time:.8f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='run_experiment',
        description='This function is made so that we can easily run configurable experiments'
    )

    parser.add_argument('-p', '--path', type=str, help='path to the dataset to use', default=None)
    parser.add_argument('-w', '--window', type=int, help='input_chunk_length', default=336)
    parser.add_argument('-b', '--batch_size', type=int, help='batch size', default=1)
    parser.add_argument('-ho', '--horizon', type=int, help='horizon', default=96)

    args = parser.parse_args()
    train_moment(
        data_path=args.path,
        window = args.window,
        batch_size= args.batch_size,
        horizon = args.horizon
    )

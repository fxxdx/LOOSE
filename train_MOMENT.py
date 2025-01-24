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

def train_moment(data_path, window, batch_size, horizon, epoch):
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

    device = 'cuda'
    # save_runs = 'results/runs/'
    # save_weights = 'results/weights/'
    # inf_time = True  # compute inference time per timeseries
    # cur_epoch = 0
    # max_epoch = 5
    # # Load the splits
    # train_set, val_set, test_set = create_splits(
    #     data_path,
    #     split_per=0.7,
    #     seed=None
    # )
    # # Uncomment for testing
    # if max_epoch == 1:
    #     train_set, val_set, test_set = train_set[:50], val_set[:10], test_set[:10]

    # Load the data
    print('----------------------------------------------------------------')
    train_dataset = InformerDataset(datapath=data_path, data_split="train", random_seed=13, forecast_horizon=horizon)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = InformerDataset(datapath=data_path, data_split="test", random_seed=13, forecast_horizon=horizon)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cur_epoch = 0
    max_epoch = epoch

    # Move the model to the GPU
    model = model.to(device)

    # Move the loss function to the GPU
    criterion = criterion.to(device)

    # Enable mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # Create a OneCycleLR scheduler
    max_lr = 1e-4
    total_steps = len(train_loader) * max_epoch
    scheduler = OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps, pct_start=0.3)

    # Gradient clipping value
    max_norm = 5.0
    starttime = datetime.datetime.now()
    while cur_epoch < max_epoch:
        losses = []
        for timeseries, forecast, input_mask in tqdm(train_loader, total=len(train_loader)):
            # Move the data to the GPU
            timeseries = timeseries.float().to(device)
            input_mask = input_mask.to(device)
            forecast = forecast.float().to(device)

            with torch.cuda.amp.autocast():
                output = model(timeseries, input_mask)

            loss = criterion(output.forecast, forecast)

            # Scales the loss for mixed precision training
            scaler.scale(loss).backward()

            # Clip gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            losses.append(loss.item())

        losses = np.array(losses)
        average_loss = np.average(losses)
        print(f"File {data_path} | Epoch {cur_epoch}: Train loss: {average_loss:.3f}")

        # Step the learning rate scheduler
        scheduler.step()
        cur_epoch += 1

        # Evaluate the model on the test split
        trues, preds, histories, losses = [], [], [], []
        model.eval()
        with torch.no_grad():
            for timeseries, forecast, input_mask in tqdm(test_loader, total=len(test_loader)):
                # Move the data to the GPU
                timeseries = timeseries.float().to(device)
                input_mask = input_mask.to(device)
                forecast = forecast.float().to(device)

                with torch.cuda.amp.autocast():
                    output = model(timeseries, input_mask)

                loss = criterion(output.forecast, forecast)
                losses.append(loss.item())

                trues.append(forecast.detach().cpu().numpy())
                preds.append(output.forecast.detach().cpu().numpy())
                histories.append(timeseries.detach().cpu().numpy())

        losses = np.array(losses)
        average_loss = np.average(losses)
        model.train()

        trues = np.concatenate(trues, axis=0)
        preds = np.concatenate(preds, axis=0)
        histories = np.concatenate(histories, axis=0)

        metrics = get_forecasting_metrics(y=trues, y_hat=preds, reduction='mean')
        endtime = datetime.datetime.now()
        totaltime = (endtime - starttime).seconds
        print(f"File {data_path} | Epoch {cur_epoch}: Test MSE: {metrics.mse:.3f} | Test MAE: {metrics.mae:.3f} | Time: {totaltime:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='run_experiment',
        description='This function is made so that we can easily run configurable experiments'
    )

    parser.add_argument('-p', '--path', type=str, help='path to the dataset to use', default=None)
    parser.add_argument('-w', '--window', type=int, help='input_chunk_length', default=336)
    parser.add_argument('-b', '--batch_size', type=int, help='batch size', default=4)
    parser.add_argument('-ho', '--horizon', type=int, help='horizon', default=60)
    parser.add_argument('-ep', '--epochs', type=int, default=15, help='number of epochs')


    args = parser.parse_args()
    train_moment(
        data_path=args.path,
        window = args.window,
        batch_size= args.batch_size,
        horizon = args.horizon,
        epoch = args.epochs
    )

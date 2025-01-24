from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def read_data(path: str, nrows=None) -> pd.DataFrame:
    """
    Read the data file and return DataFrame.
    According to the provided file path, read the data file and return the corresponding DataFrame.
    :param path: The path to the data file.
    :return:  The DataFrame of the content of the data file.
    """
    data = pd.read_csv(path)
    label_exists = "label" in data["cols"].values

    all_points = data.shape[0]

    columns = data.columns

    if columns[0] == "date":
        n_points = data.iloc[:, 2].value_counts().max()
        print(data.iloc[:, 2].value_counts())
    else:
        n_points = data.iloc[:, 1].value_counts().max()
    print("------------------n_points-------------------------", n_points)
    print(all_points)
    is_univariate = n_points == all_points

    n_cols = all_points // n_points
    df = pd.DataFrame()

    cols_name = data["cols"].unique()

    if columns[0] == "date" and not is_univariate:
        df["date"] = data.iloc[:n_points, 0]
        col_data = {
            cols_name[j]: data.iloc[j * n_points: (j + 1) * n_points, 1].tolist()
            for j in range(n_cols)
        }
        df = pd.concat([df, pd.DataFrame(col_data)], axis=1)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

    elif columns[0] != "date" and not is_univariate:
        col_data = {
            cols_name[j]: data.iloc[j * n_points: (j + 1) * n_points, 0].tolist()
            for j in range(n_cols)
        }
        df = pd.concat([df, pd.DataFrame(col_data)], axis=1)

    elif columns[0] == "date" and is_univariate:
        df["date"] = data.iloc[:, 0]
        df[cols_name[0]] = data.iloc[:, 1]

        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

    else:
        df[cols_name[0]] = data.iloc[:, 0]

    if label_exists:
        # Get the column name of the last column
        last_col_name = df.columns[-1]
        # Renaming the last column as "label"
        df.rename(columns={last_col_name: "label"}, inplace=True)

    if nrows is not None and isinstance(nrows, int) and df.shape[0] >= nrows:
        df = df.iloc[:nrows, :]
    print(df)
    return df

class InformerDataset:
    def __init__(
        self,
        datapath,
        forecast_horizon: Optional[int] = 192,
        data_split: str = "train",
        data_stride_len: int = 1,
        task_name: str = "forecasting",
        random_seed: int = 42,
    ):
        """
        Parameters
        ----------
        forecast_horizon : int
            Length of the prediction sequence.
        data_split : str
            Split of the dataset, 'train' or 'test'.
        data_stride_len : int
            Stride length when generating consecutive
            time series windows.
        task_name : str
            The task that the dataset is used for. One of
            'forecasting', or  'imputation'.
        random_seed : int
            Random seed for reproducibility.
        """

        self.seq_len = 336
        self.forecast_horizon = forecast_horizon
        # self.full_file_path_and_name = "/home/fanxiaoxuan/tmp/moment/data/ETTh1.csv"
        self.full_file_path_and_name = datapath
        self.data_split = data_split
        self.data_stride_len = data_stride_len
        self.task_name = task_name
        self.random_seed = random_seed

        # Read data
        self._read_data()

    def _get_borders(self, length):
        # n_train = 12 * 30 * 24
        # n_val = 4 * 30 * 24
        # n_test = 4 * 30 * 24

        n_train = int(0.6 * length)
        n_val = int(0.2 * length)
        n_test = int(0.2 * length)

        train_end = n_train
        val_end = n_train + n_val
        test_start = val_end - self.seq_len
        test_end = test_start + n_test + self.seq_len

        train = slice(0, train_end)
        test = slice(test_start, test_end)
        print(train)
        print(test)
        return train, test

    def _read_data(self):
        self.scaler = StandardScaler()
        print("-------------path-------------------")
        print(self.full_file_path_and_name)
        df = read_data(self.full_file_path_and_name)
        self.length_timeseries_original = df.shape[0]
        # print("self.length_timeseries_original:  ", self.length_timeseries_original)
        self.n_channels = df.shape[1] - 1
        # print(df.columns)
        # print(df.shape)

        # df.drop(columns=["date"], inplace=True)
        df = df.infer_objects(copy=False).interpolate(method="cubic")

        data_splits = self._get_borders(self.length_timeseries_original)

        train_data = df[data_splits[0]]
        self.scaler.fit(train_data.values)
        df = self.scaler.transform(df.values)

        if self.data_split == "train":
            self.data = df[data_splits[0], :]
        if self.data_split == "test":
            self.data = df[data_splits[1], :]
        elif self.data_split == "no":
            self.data = df

        self.length_timeseries = self.data.shape[0]

    def __getitem__(self, index):
        seq_start = self.data_stride_len * index
        seq_end = seq_start + self.seq_len
        input_mask = np.ones(self.seq_len)

        if self.task_name == "forecasting":
            pred_end = seq_end + self.forecast_horizon

            if pred_end > self.length_timeseries:
                pred_end = self.length_timeseries
                seq_end = seq_end - self.forecast_horizon
                seq_start = seq_end - self.seq_len

            timeseries = self.data[seq_start:seq_end, :].T
            forecast = self.data[seq_end:pred_end, :].T
            # print("timeseries", timeseries.shape)
            # print("forecast", forecast.shape)
            return timeseries, forecast, input_mask

        elif self.task_name == "imputation":
            if seq_end > self.length_timeseries:
                seq_end = self.length_timeseries
                seq_end = seq_end - self.seq_len

            timeseries = self.data[seq_start:seq_end, :].T

            return timeseries, input_mask

    def __len__(self):
        if self.task_name == "imputation":
            return (self.length_timeseries - self.seq_len) // self.data_stride_len + 1
        elif self.task_name == "forecasting":
            return (
                self.length_timeseries - self.seq_len - self.forecast_horizon
            ) // self.data_stride_len + 1

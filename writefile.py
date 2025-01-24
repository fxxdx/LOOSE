import os
import pandas as pd
import numpy as np
from pathlib import Path
import math
import argparse
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

def write_file(path, readpath, metric, seq_len, horizon_len):

    target_path = os.path.join(path, 'horizon_' + str(horizon_len), 'lookback_' + str(seq_len))

    for forcast_model in forcast_names:
        data_list = []
        dataname_list= []

        for i in os.listdir(readpath):
            if i == 'bus': continue
            dataname_list.append(i+'.csv')
            findpath = os.path.join(readpath, i, forcast_model)
            min_metrics = math.inf
            if os.path.exists(findpath)!=False:
                for file in os.listdir(findpath):
                    if file.startswith('test_report_' + str(horizon_len) + '_' + str(seq_len)):
                        df = pd.read_csv(os.path.join(readpath, i, forcast_model, file))
                        # data = np.array(df)[0, -1]    #mae
                        if metric=='mae': data = np.array(df)[0, -1]    #mae
                        elif metric=='mse': data = np.array(df)[1, -1]    #mse
                        elif metric == 'rmse':
                            data = np.array(df)[2, -1]  # rmse
                        min_metrics = min(min_metrics, data)

            data_list.append(min_metrics)
        data_list = np.array(data_list)[:, np.newaxis]
        dataname_list = np.array(dataname_list)[:, np.newaxis]
        print(data_list.shape)
        print(dataname_list.shape)
        finaldata = np.concatenate([dataname_list, data_list], axis=1)
        Path(os.path.join(target_path, forcast_model)).mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(data=finaldata, columns=['datafile', forcast_model])

        df.to_csv(os.path.join(target_path, forcast_model, metric + '.csv'), index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='run_experiment',
        description='This function is made so that we can easily run configurable experiments'
    )

    parser.add_argument('-p', '--path', type=str, help='path to the dataset to use', default='dataset/metric')
    parser.add_argument('-r', '--readpath', type=str, help='input_chunk_length', default='result')
    parser.add_argument('-m', '--metric', type=str, help='metric', default='mae')
    parser.add_argument('-w', '--window', type=int, help='input_chunk_length', default=336)
    parser.add_argument('-ho', '--horizon', type=int, help='horizon', default=60)

    args = parser.parse_args()
    write_file(
        path=args.path,
        readpath = args.readpath,
        metric= args.metric,
        seq_len=args.window,
        horizon_len=args.horizon
    )

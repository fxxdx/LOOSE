python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":60}' --model-name "time_series_library.Crossformer" --model-hyper-params '{"batch_size": 4, "d_ff": 128, "d_model": 64, "dropout": 0.2, "e_layers": 3, "factor": 10, "lr": 0.001, "n_headers": 2, "num_epochs": 20, "horizon": 60, "seg_len": 12, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 1  --num-workers 1  --timeout 60000  --save-path "PEMS08/Crossformer"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":60}' --model-name "time_series_library.DLinear" --model-hyper-params '{"batch_size": 8, "lr": 0.005, "horizon": 60, "seq_len": 336, "d_ff": 256, "d_model": 128}' --adapter "transformer_adapter"  --gpus 1  --num-workers 1  --timeout 60000  --save-path "PEMS08/DLinear"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":60}' --model-name "time_series_library.FEDformer" --model-hyper-params '{"d_ff": 512, "d_model": 256, "horizon": 60, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 1  --num-workers 1  --timeout 60000  --save-path "PEMS08/FEDformer"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":60}' --model-name "time_series_library.FiLM" --model-hyper-params '{"batch_size": 2, "dropout": 0.05, "factor": 3, "moving_avg": 24, "num_epochs": 15, "patience": 15, "horizon": 60, "seq_len": 336, "d_ff": 256, "d_model": 128}' --adapter "transformer_adapter"  --gpus 1  --num-workers 1  --timeout 60000  --save-path "PEMS08/FiLM"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":60}' --model-name "time_series_library.Informer" --model-hyper-params '{"d_ff": 512, "d_model": 256, "horizon": 60, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 1  --num-workers 1  --timeout 60000  --save-path "PEMS08/Informer"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":60}' --model-name "time_series_library.Linear" --model-hyper-params '{"batch_size": 8, "lr": 0.005, "horizon": 60, "seq_len": 336, "d_ff": 2048, "d_model": 512}' --adapter "transformer_adapter"  --gpus 1  --num-workers 1  --timeout 60000  --save-path "PEMS08/Linear"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":60}' --model-name "time_series_library.MICN" --model-hyper-params '{"d_ff": 512, "d_model": 256, "horizon": 60, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 1  --num-workers 1  --timeout 60000  --save-path "PEMS08/MICN"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":60}' --model-name "time_series_library.NLinear" --model-hyper-params '{"batch_size": 8, "lr": 0.005, "horizon": 60, "seq_len": 336, "d_ff": 256, "d_model": 128}' --adapter "transformer_adapter"  --gpus 1  --num-workers 1  --timeout 60000  --save-path "PEMS08/NLinear"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":60}' --model-name "time_series_library.Nonstationary_Transformer" --model-hyper-params '{"batch_size": 8, "dropout": 0.05, "factor": 3, "p_hidden_dims": [128, 128], "p_hidden_layers": 2, "horizon": 60, "seq_len": 336, "d_ff": 2048, "d_model": 512}' --adapter "transformer_adapter"  --gpus 1  --num-workers 1  --timeout 60000  --save-path "PEMS08/Nonstationary_Transformer"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":60}' --model-name "time_series_library.PatchTST" --model-hyper-params '{"batch_size": 8, "d_ff": 256, "d_model": 128, "horizon": 60, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 1  --num-workers 1  --timeout 60000  --save-path "PEMS08/PatchTST"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":60}' --model-name "darts.RegressionModel" --model-hyper-params '{"output_chunk_length": 1}' --gpus 0  --num-workers 1  --timeout 60000  --save-path "PEMS08/regressionmodel"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":60}' --model-name "darts.TCNModel" --model-hyper-params '{"input_chunk_length": 336, "n_epochs": 10, "output_chunk_length": 60}' --gpus 0  --num-workers 1  --timeout 60000  --save-path "PEMS08/TCN"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":60}' --model-name "time_series_library.TimesNet" --model-hyper-params '{"batch_size": 8, "d_ff": 256, "d_model": 128, "factor": 3, "horizon": 60, "seq_len": 336, "top_k": 5}' --adapter "transformer_adapter"  --gpus 1  --num-workers 1  --timeout 60000  --save-path "PEMS08/TimesNet"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":60}' --model-name "time_series_library.Triformer" --model-hyper-params '{"batch_size": 8, "d_ff": 64, "d_model": 32, "horizon": 60, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 1  --num-workers 1  --timeout 60000  --save-path "PEMS08/Triformer"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":60}' --model-name "self_impl.VAR_model" --gpus 1  --num-workers 1  --timeout 60000  --save-path "PEMS08/VAR"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":60}' --model-name "darts.RNNModel" --model-hyper-params '{"input_chunk_length": 24}' --gpus 0  --num-workers 1  --timeout 60000  --save-path "PEMS08/RNN"

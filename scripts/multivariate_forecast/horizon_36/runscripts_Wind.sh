#python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "Wind.csv" --strategy-args '{"horizon":36}' --model-name "time_series_library.Crossformer" --model-hyper-params '{"batch_size": 2, "d_ff": 256, "d_model": 128, "dropout": 0.2, "e_layers": 3, "factor": 10, "lr": 0.0001, "n_headers": 4, "num_epochs": 20, "horizon":36, "seg_len": 12, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 2  --num-workers 1  --timeout 60000  --save-path "Wind/Crossformer"
#
#python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "Wind.csv" --strategy-args '{"horizon":36}' --model-name "time_series_library.DLinear" --model-hyper-params '{"batch_size": 2, "d_ff": 512, "d_model": 256, "lr": 0.001, "horizon":36, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 2  --num-workers 1  --timeout 60000  --save-path "Wind/DLinear"
#
#python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "Wind.csv" --strategy-args '{"horizon":36}' --model-name "time_series_library.FEDformer" --model-hyper-params '{"batch_size": 2, "dropout": 0.05, "factor": 3, "moving_avg": 25, "horizon":36, "seq_len": 336, "d_ff": 2048, "d_model": 512}' --adapter "transformer_adapter"  --gpus 2  --num-workers 1  --timeout 60000  --save-path "Wind/FEDformer"
#
#python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "Wind.csv" --strategy-args '{"horizon":36}' --model-name "time_series_library.FiLM" --model-hyper-params '{"batch_size": 2, "d_ff": 64, "d_model": 32, "dropout": 0.05, "factor": 3, "lr": 0.001, "moving_avg": 25, "num_epochs": 15, "patience": 15, "horizon":36, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 2  --num-workers 1  --timeout 60000  --save-path "Wind/FiLM"
#
#python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "Wind.csv" --strategy-args '{"horizon":36}' --model-name "time_series_library.Informer" --model-hyper-params '{"batch_size": 2, "d_ff": 256, "d_model": 128, "horizon":36, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 2  --num-workers 1  --timeout 60000  --save-path "Wind/Informer"
#
#python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "Wind.csv" --strategy-args '{"horizon":36}' --model-name "time_series_library.Linear" --model-hyper-params '{"batch_size": 2, "d_ff": 512, "d_model": 256, "lr": 0.01, "horizon":36, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 2  --num-workers 1  --timeout 60000  --save-path "Wind/Linear"
#
#python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "Wind.csv" --strategy-args '{"horizon":36}' --model-name "time_series_library.MICN" --model-hyper-params '{"batch_size": 2, "d_ff": 64, "d_model": 32, "dropout": 0.05, "lr": 0.001, "moving_avg": 25, "num_epochs": 15, "horizon":36, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 2  --num-workers 1  --timeout 60000  --save-path "Wind/MICN"
#
#python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "Wind.csv" --strategy-args '{"horizon":36}' --model-name "time_series_library.NLinear" --model-hyper-params '{"batch_size": 2, "d_ff": 512, "d_model": 256, "lr": 0.001, "horizon":36, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 2  --num-workers 1  --timeout 60000  --save-path "Wind/NLinear"
#
#python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "Wind.csv" --strategy-args '{"horizon":36}' --model-name "time_series_library.Nonstationary_Transformer" --model-hyper-params '{"batch_size": 2, "dropout": 0.05, "factor": 3, "p_hidden_dims": [256, 256], "p_hidden_layers": 2, "horizon":36, "seq_len": 336, "d_ff": 2048, "d_model": 512}' --adapter "transformer_adapter"  --gpus 2  --num-workers 1  --timeout 60000  --save-path "Wind/Nonstationary_Transformer"
#
#python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "Wind.csv" --strategy-args '{"horizon":36}' --model-name "time_series_library.PatchTST" --model-hyper-params '{"batch_size": 2, "d_ff": 512, "d_model": 256, "dropout": 0.2, "e_layers": 3, "n_headers": 16, "num_epochs": 100, "patience": 20, "horizon":36, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 2  --num-workers 1  --timeout 60000  --save-path "Wind/PatchTST"
#
python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "Wind.csv" --strategy-args '{"horizon":36}' --model-name "darts.RegressionModel" --model-hyper-params '{"output_chunk_length": 36}' --gpus 1  --num-workers 1  --timeout 60000  --save-path "Wind/regressionmodel"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "Wind.csv" --strategy-args '{"horizon":36}' --model-name "darts.RNNModel" --model-hyper-params '{"input_chunk_length": 24}' --gpus 1  --num-workers 1  --timeout 60000  --save-path "Wind/RNN"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "Wind.csv" --strategy-args '{"horizon":36}' --model-name "darts.TCNModel" --model-hyper-params '{"input_chunk_length": 336, "n_epochs": 100, "output_chunk_length": 36}' --gpus 1  --num-workers 1  --timeout 60000  --save-path "Wind/TCN"
#
#python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "Wind.csv" --strategy-args '{"horizon":36}' --model-name "time_series_library.TimesNet" --model-hyper-params '{"batch_size": 2, "d_ff": 64, "d_model": 32, "factor": 3, "horizon":36, "seq_len": 336, "top_k": 5}' --adapter "transformer_adapter"  --gpus 2  --num-workers 1  --timeout 60000  --save-path "Wind/TimesNet"
#
#python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "Wind.csv" --strategy-args '{"horizon":36}' --model-name "time_series_library.Triformer" --model-hyper-params '{"batch_size": 2, "d_ff": 64, "d_model": 32, "horizon":36, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 2  --num-workers 1  --timeout 60000  --save-path "Wind/Triformer"
#
#python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "Wind.csv" --strategy-args '{"horizon":36}' --model-name "self_impl.VAR_model" --gpus 2  --num-workers 1  --timeout 60000  --save-path "Wind/VAR"

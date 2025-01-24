#python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon":48}' --model-name "time_series_library.Crossformer" --model-hyper-params '{"batch_size": 2, "d_ff": 32, "d_model": 16, "dropout": 0.1, "e_layers": 2, "factor": 3, "lr": 0.0001, "n_headers": 16, "num_epochs": 20, "horizon":48, "seg_len": 6, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 2  --num-workers 1  --timeout 60000  --save-path "Wike2000/Crossformer"
#
#python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon":48}' --model-name "time_series_library.DLinear" --model-hyper-params '{"batch_size": 2, "d_ff": 64, "d_model": 32, "lr": 0.01, "horizon":48, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 2  --num-workers 1  --timeout 60000  --save-path "Wike2000/DLinear"
#
#python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon":48}' --model-name "time_series_library.FEDformer" --model-hyper-params '{"batch_size": 2, "d_ff": 64, "d_model": 32, "dropout": 0.05, "factor": 3, "moving_avg": 25, "horizon":48, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 2  --num-workers 1  --timeout 60000  --save-path "Wike2000/FEDformer"
#
#python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon":48}' --model-name "time_series_library.FiLM" --model-hyper-params '{"batch_size": 1, "dropout": 0.05, "factor": 3, "lr": 0.001, "moving_avg": 25, "num_epochs": 60, "patience": 60, "horizon":48, "seq_len": 336, "d_ff": 64, "d_model": 32}' --adapter "transformer_adapter"  --gpus 1  --num-workers 1  --timeout 60000  --save-path "Wike2000/FiLM"
#
#python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon":48}' --model-name "time_series_library.Informer" --model-hyper-params '{"batch_size": 2, "d_ff": 64, "d_model": 32, "horizon":48, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 2  --num-workers 1  --timeout 60000  --save-path "Wike2000/Informer"
#
#python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon":48}' --model-name "time_series_library.Linear" --model-hyper-params '{"batch_size": 2, "d_ff": 64, "d_model": 32, "lr": 0.01, "horizon":48, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 2  --num-workers 1  --timeout 60000  --save-path "Wike2000/Linear"
#
#python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon":48}' --model-name "time_series_library.MICN" --model-hyper-params '{"batch_size": 2, "conv_kernel": [18, 12], "d_ff": 256, "d_model": 128, "dropout": 0.05, "lr": 0.001, "moving_avg": 25, "num_epochs": 15, "horizon":48, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 2  --num-workers 1  --timeout 60000  --save-path "Wike2000/MICN"
#
#python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon":48}' --model-name "time_series_library.NLinear" --model-hyper-params '{"batch_size": 2, "d_ff": 64, "d_model": 32, "lr": 0.01, "horizon":48, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 2  --num-workers 1  --timeout 60000  --save-path "Wike2000/NLinear"
#
#python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon":48}' --model-name "time_series_library.Nonstationary_Transformer" --model-hyper-params '{"batch_size": 2, "dropout": 0.05, "factor": 3, "p_hidden_dims": [8, 8], "p_hidden_layers": 2, "horizon":48, "seq_len": 336, "d_ff": 2048, "d_model": 512}' --adapter "transformer_adapter"  --gpus 2  --num-workers 1  --timeout 60000  --save-path "Wike2000/Nonstationary_Transformer"
#
#python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon":48}' --model-name "time_series_library.PatchTST" --model-hyper-params '{"batch_size": 1, "d_ff": 16, "d_model": 8, "dropout": 0.3, "e_layers": 3, "lr": 0.0025, "n_headers": 4, "num_epochs": 100, "patch_len": 24, "patience": 100, "horizon":48, "seq_len": 336, "stride": 2}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Wike2000/PatchTST"
#
python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon":48}' --model-name "darts.RegressionModel" --model-hyper-params '{"output_chunk_length": 1}' --gpus 0 --num-workers 1  --timeout 60000  --save-path "Wike2000/regressionmodel"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon":48}' --model-name "darts.TCNModel" --model-hyper-params '{"input_chunk_length": 336, "n_epochs": 100, "output_chunk_length": 48}' --gpus 0  --num-workers 1  --timeout 60000  --save-path "Wike2000/TCN"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon":48}' --model-name "darts.RNNModel" --gpus 0  --model-hyper-params '{"input_chunk_length": 24}' --num-workers 1  --timeout 60000  --save-path "Wike2000/RNN"
#
#python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon":48}' --model-name "time_series_library.TimesNet" --model-hyper-params '{"batch_size": 2, "d_ff": 128, "d_model": 64, "factor": 3, "horizon":48, "seq_len": 336, "top_k": 5}' --adapter "transformer_adapter"  --gpus 2  --num-workers 1  --timeout 60000  --save-path "Wike2000/TimesNet"
#
#python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon":48}' --model-name "time_series_library.Triformer" --model-hyper-params '{"batch_size": 2, "d_ff": 16, "d_model": 8, "horizon":48, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 2  --num-workers 1  --timeout 60000  --save-path "Wike2000/Triformer"
#
#python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon":48}' --model-name "self_impl.VAR_model" --gpus 2  --num-workers 1  --timeout 60000  --save-path "Wike2000/VAR"

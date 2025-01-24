python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "Electricity.csv" --strategy-args '{"horizon":48}' --model-name "time_series_library.Crossformer" --model-hyper-params '{"batch_size": 2, "d_ff": 128, "d_model": 64, "dropout": 0.2, "e_layers": 3, "factor": 10, "lr": 0.0005, "n_headers": 2, "num_epochs": 20, "horizon": 48, "seg_len": 12, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Electricity/Crossformer"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "Electricity.csv" --strategy-args '{"horizon":48}' --model-name "time_series_library.DLinear" --model-hyper-params '{"batch_size": 2, "lr": 0.005, "horizon": 48, "seq_len": 336, "d_ff": 2048, "d_model": 512}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Electricity/DLinear"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "Electricity.csv" --strategy-args '{"horizon":48}' --model-name "time_series_library.FEDformer" --model-hyper-params '{"batch_size": 2, "d_ff": 512, "d_model": 256, "dropout": 0.05, "factor": 3, "moving_avg": 25, "horizon": 48, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Electricity/FEDformer"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "Electricity.csv" --strategy-args '{"horizon":48}' --model-name "time_series_library.FiLM" --model-hyper-params '{"batch_size": 2, "dropout": 0.05, "factor": 3, "lr": 0.001, "moving_avg": 25, "num_epochs": 20, "patience": 20, "horizon": 48, "seq_len": 336, "d_ff": 512, "d_model": 256}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Electricity/FiLM"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "Electricity.csv" --strategy-args '{"horizon":48}' --model-name "time_series_library.Informer" --model-hyper-params '{"batch_size": 2, "factor": 3, "horizon": 48, "seq_len": 336, "d_ff": 2048, "d_model": 512}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Electricity/Informer"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "Electricity.csv" --strategy-args '{"horizon":48}' --model-name "time_series_library.Linear" --model-hyper-params '{"batch_size": 2, "lr": 0.005, "horizon": 48, "seq_len": 336, "d_ff": 2048, "d_model": 512}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Electricity/Linear"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "Electricity.csv" --strategy-args '{"horizon":48}' --model-name "time_series_library.MICN" --model-hyper-params '{"batch_size": 2, "d_model": 512, "dropout": 0.05, "lr": 0.001, "moving_avg": 24, "num_epochs": 15, "horizon": 48, "seq_len": 336, "d_ff": 2048}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Electricity/MICN"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "Electricity.csv" --strategy-args '{"horizon":48}' --model-name "time_series_library.NLinear" --model-hyper-params '{"batch_size": 2, "lr": 0.005, "horizon": 48, "seq_len": 336, "d_ff": 2048, "d_model": 512}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Electricity/NLinear"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "Electricity.csv" --strategy-args '{"horizon":48}' --model-name "time_series_library.Nonstationary_Transformer" --model-hyper-params '{"batch_size": 2, "dropout": 0.05, "factor": 3, "p_hidden_dims": [256, 256], "p_hidden_layers": 2, "horizon": 48, "seq_len": 336, "d_ff": 2048, "d_model": 512}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Electricity/Nonstationary_Transformer"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "Electricity.csv" --strategy-args '{"horizon":48}' --model-name "time_series_library.PatchTST" --model-hyper-params '{"batch_size": 2, "d_ff": 512, "d_model": 256, "lr": 0.001, "horizon": 48, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Electricity/PatchTST"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "Electricity.csv" --strategy-args '{"horizon":48}' --model-name "darts.RegressionModel"  --model-hyper-params '{"output_chunk_length": 1}' --gpus 0  --num-workers 1  --timeout 60000  --save-path "Electricity/regressionmodel"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "Electricity.csv" --strategy-args '{"horizon":48}' --model-name "darts.RNNModel" --model-hyper-params '{"input_chunk_length": 24}' --gpus 0  --num-workers 1  --timeout 60000  --save-path "Electricity/RNN"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "Electricity.csv" --strategy-args '{"horizon":48}' --model-name "darts.TCNModel" --model-hyper-params '{"input_chunk_length": 336, "n_epochs": 10, "output_chunk_length": 48}' --gpus 0  --num-workers 1  --timeout 60000  --save-path "Electricity/TCN"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "Electricity.csv" --strategy-args '{"horizon":48}' --model-name "time_series_library.TimesNet" --model-hyper-params '{"batch_size": 2, "d_ff": 512, "d_model": 256, "lr": 0.0005, "num_epochs": 20, "horizon": 48, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Electricity/TimesNet"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "Electricity.csv" --strategy-args '{"horizon":48}' --model-name "time_series_library.Triformer" --model-hyper-params '{"batch_size": 2, "d_ff": 64, "d_model": 32, "horizon": 48, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Electricity/Triformer"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "Electricity.csv" --strategy-args '{"horizon":48}' --model-name "self_impl.VAR_model" --gpus 0  --num-workers 1  --timeout 60000  --save-path "Electricity/VAR"

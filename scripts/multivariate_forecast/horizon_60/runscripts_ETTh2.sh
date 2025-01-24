python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "ETTh2.csv" --strategy-args '{"horizon":60}' --model-name "time_series_library.Crossformer" --model-hyper-params '{"batch_size": 8, "d_ff": 128, "d_model": 32, "horizon": 60, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ETTh2/Crossformer"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "ETTh2.csv" --strategy-args '{"horizon":60}' --model-name "time_series_library.DLinear" --model-hyper-params '{"batch_size": 8, "factor": 3, "horizon": 60, "seq_len": 336, "d_ff": 2048, "d_model": 512}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ETTh2/DLinear"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "ETTh2.csv" --strategy-args '{"horizon":60}' --model-name "time_series_library.FEDformer" --model-hyper-params '{"batch_size": 8, "d_ff": 512, "d_model": 128, "horizon": 60, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ETTh2/FEDformer"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "ETTh2.csv" --strategy-args '{"horizon":60}' --model-name "time_series_library.FiLM" --model-hyper-params '{"batch_size": 8, "dropout": 0.05, "factor": 3, "lr": 0.001, "moving_avg": 25, "num_epochs": 3, "patience": 3, "horizon": 60, "seq_len": 336, "d_ff": 2048, "d_model": 512}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ETTh2/FiLM"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "ETTh2.csv" --strategy-args '{"horizon":60}' --model-name "time_series_library.Informer" --model-hyper-params '{"batch_size": 8, "d_ff": 128, "d_model": 32, "horizon": 60, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ETTh2/Informer"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "ETTh2.csv" --strategy-args '{"horizon":60}' --model-name "time_series_library.Linear" --model-hyper-params '{"batch_size": 8, "lr": 0.005, "horizon": 60, "seq_len": 336, "d_ff": 2048, "d_model": 512}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ETTh2/Linear"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "ETTh2.csv" --strategy-args '{"horizon":60}' --model-name "time_series_library.MICN" --model-hyper-params '{"batch_size": 8, "d_ff": 512, "d_model": 128, "horizon": 60, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ETTh2/MICN"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "ETTh2.csv" --strategy-args '{"horizon":60}' --model-name "time_series_library.NLinear" --model-hyper-params '{"batch_size": 8, "lr": 0.005, "horizon": 60, "seq_len": 336, "d_ff": 2048, "d_model": 512}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ETTh2/NLinear"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "ETTh2.csv" --strategy-args '{"horizon":60}' --model-name "time_series_library.Nonstationary_Transformer" --model-hyper-params '{"batch_size": 8, "d_ff": 512, "d_model": 128, "p_hidden_dims": [256, 256], "p_hidden_layers": 2, "horizon": 60, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ETTh2/Nonstationary_Transformer"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "ETTh2.csv" --strategy-args '{"horizon":60}' --model-name "time_series_library.PatchTST" --model-hyper-params '{"batch_size": 8, "d_ff": 128, "d_model": 32, "horizon": 60, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ETTh2/PatchTST"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "ETTh2.csv" --strategy-args '{"horizon":60}' --model-name "darts.RegressionModel" --model-hyper-params '{"output_chunk_length": 60}' --gpus 0  --num-workers 1  --timeout 60000  --save-path "ETTh2/regressionmodel"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "ETTh2.csv" --strategy-args '{"horizon":60}' --model-name "darts.RNNModel" --model-hyper-params '{"input_chunk_length": 24}' --gpus 0  --num-workers 1  --timeout 60000  --save-path "ETTh2/RNN"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "ETTh2.csv" --strategy-args '{"horizon":60}' --model-name "darts.TCNModel" --model-hyper-params '{"input_chunk_length": 336, "n_epochs": 10, "output_chunk_length": 60}' --gpus 0  --num-workers 1  --timeout 60000  --save-path "ETTh2/TCN"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "ETTh2.csv" --strategy-args '{"horizon":60}' --model-name "time_series_library.TimesNet" --model-hyper-params '{"batch_size": 8, "d_ff": 128, "d_model": 32, "horizon": 60, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ETTh2/TimesNet"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "ETTh2.csv" --strategy-args '{"horizon":60}' --model-name "time_series_library.Triformer" --model-hyper-params '{"batch_size": 8, "d_ff": 64, "d_model": 32, "horizon": 60, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ETTh2/Triformer"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "ETTh2.csv" --strategy-args '{"horizon":60}' --model-name "self_impl.VAR_model" --gpus 0  --num-workers 1  --timeout 60000  --save-path "ETTh2/VAR"


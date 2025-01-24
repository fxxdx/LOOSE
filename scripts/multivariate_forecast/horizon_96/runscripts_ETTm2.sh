python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "ETTm2.csv" --strategy-args '{"horizon":96}' --model-name "time_series_library.Crossformer" --model-hyper-params '{"batch_size": 4, "d_ff": 64, "d_model": 32, "horizon": 96, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 1  --num-workers 1  --timeout 60000  --save-path "ETTm2/Crossformer"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "ETTm2.csv" --strategy-args '{"horizon":96}' --model-name "time_series_library.DLinear" --model-hyper-params '{"batch_size": 4, "d_ff": 512, "d_model": 256, "horizon": 96, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 1  --num-workers 1  --timeout 60000  --save-path "ETTm2/DLinear"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "ETTm2.csv" --strategy-args '{"horizon":96}' --model-name "time_series_library.FEDformer" --model-hyper-params '{"batch_size": 4, "d_ff": 256, "d_model": 128, "horizon": 96, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 1  --num-workers 1  --timeout 60000  --save-path "ETTm2/FEDformer"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "ETTm2.csv" --strategy-args '{"horizon":96}' --model-name "time_series_library.FiLM" --model-hyper-params '{"batch_size": 4, "dropout": 0.05, "lr": 0.001, "moving_avg": 25, "num_epochs": 30, "patience": 30, "horizon": 96, "seq_len": 336, "d_ff": 2048, "d_model": 512}' --adapter "transformer_adapter"  --gpus 1  --num-workers 1  --timeout 60000  --save-path "ETTm2/FiLM"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "ETTm2.csv" --strategy-args '{"horizon":96}' --model-name "time_series_library.Informer" --model-hyper-params '{"batch_size": 4, "d_ff": 64, "d_model": 32, "horizon": 96, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 1  --num-workers 1  --timeout 60000  --save-path "ETTm2/Informer"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "ETTm2.csv" --strategy-args '{"horizon":96}' --model-name "time_series_library.Linear" --model-hyper-params '{"batch_size": 4, "d_ff": 512, "d_model": 256, "lr": 0.01, "horizon": 96, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 1  --num-workers 1  --timeout 60000  --save-path "ETTm2/Linear"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "ETTm2.csv" --strategy-args '{"horizon":96}' --model-name "time_series_library.MICN" --model-hyper-params '{"batch_size": 4, "d_ff": 64, "d_model": 32, "horizon": 96, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 1  --num-workers 1  --timeout 60000  --save-path "ETTm2/MICN"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "ETTm2.csv" --strategy-args '{"horizon":96}' --model-name "time_series_library.NLinear" --model-hyper-params '{"batch_size": 4, "d_ff": 512, "d_model": 256, "lr": 0.01, "horizon": 96, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 1  --num-workers 1  --timeout 60000  --save-path "ETTm2/NLinear"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "ETTm2.csv" --strategy-args '{"horizon":96}' --model-name "time_series_library.Nonstationary_Transformer" --model-hyper-params '{"batch_size": 4, "d_ff": 256, "d_model": 128, "p_hidden_dims": [256, 256, 256, 256], "p_hidden_layers": 4, "horizon": 96, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 1  --num-workers 1  --timeout 60000  --save-path "ETTm2/Nonstationary_Transformer"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "ETTm2.csv" --strategy-args '{"horizon":96}' --model-name "time_series_library.PatchTST" --model-hyper-params '{"batch_size": 4, "d_ff": 64, "d_model": 32, "horizon": 96, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 1  --num-workers 1  --timeout 60000  --save-path "ETTm2/PatchTST"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "ETTm2.csv" --strategy-args '{"horizon":96}' --model-name "darts.RegressionModel" --model-hyper-params '{"output_chunk_length": 96}' --gpus 0  --num-workers 1  --timeout 60000  --save-path "ETTm2/regressionmodel"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "ETTm2.csv" --strategy-args '{"horizon":96}' --model-name "darts.RNNModel" --model-hyper-params '{"input_chunk_length": 24}' --gpus 0  --num-workers 1  --timeout 60000  --save-path "ETTm2/RNN"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "ETTm2.csv" --strategy-args '{"horizon":96}' --model-name "darts.TCNModel" --model-hyper-params '{"input_chunk_length": 336, "n_epochs": 10, "output_chunk_length": 96}' --gpus 0  --num-workers 1  --timeout 60000  --save-path "ETTm2/TCN"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "ETTm2.csv" --strategy-args '{"horizon":96}' --model-name "time_series_library.TimesNet" --model-hyper-params '{"batch_size": 4, "d_ff": 64, "d_model": 32, "horizon": 96, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 1  --num-workers 1  --timeout 60000  --save-path "ETTm2/TimesNet"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "ETTm2.csv" --strategy-args '{"horizon":96}' --model-name "time_series_library.Triformer" --model-hyper-params '{"batch_size": 4, "d_ff": 64, "d_model": 32, "horizon": 96, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 1  --num-workers 1  --timeout 60000  --save-path "ETTm2/Triformer"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_multi.json" --data-name-list "ETTm2.csv" --strategy-args '{"horizon":96}' --model-name "self_impl.VAR_model" --gpus 1  --num-workers 1  --timeout 60000  --save-path "ETTm2/VAR"

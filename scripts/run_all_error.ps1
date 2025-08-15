# LSTM
python -m src.error_analysis --pred_csv experiments/power_lstm/test_predictions.csv   --out_dir experiments/power_lstm/error_analysis
python -m src.error_analysis --pred_csv experiments/traffic_lstm/test_predictions.csv --out_dir experiments/traffic_lstm/error_analysis
python -m src.error_analysis --pred_csv experiments/weather_lstm/test_predictions.csv --out_dir experiments/weather_lstm/error_analysis

# Transformer
python -m src.error_analysis --pred_csv experiments/power_transformer/test_predictions.csv   --out_dir experiments/power_transformer/error_analysis
python -m src.error_analysis --pred_csv experiments/traffic_transformer/test_predictions.csv --out_dir experiments/traffic_transformer/error_analysis
python -m src.error_analysis --pred_csv experiments/weather_transformer/test_predictions.csv --out_dir experiments/weather_transformer/error_analysis

# Hybrid
python -m src.error_analysis --pred_csv experiments/power_hybrid/test_predictions.csv   --out_dir experiments/power_hybrid/error_analysis
python -m src.error_analysis --pred_csv experiments/traffic_hybrid/test_predictions.csv --out_dir experiments/traffic_hybrid/error_analysis
python -m src.error_analysis --pred_csv experiments/weather_hybrid/test_predictions.csv --out_dir experiments/weather_hybrid/error_analysis

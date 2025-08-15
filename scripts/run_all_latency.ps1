# LSTM
python -m src.latency --config configs/power.yaml --ckpt experiments/power_lstm/best.ckpt
python -m src.latency --config configs/traffic.yaml --ckpt experiments/traffic_lstm/best.ckpt
python -m src.latency --config configs/weather.yaml --ckpt experiments/weather_lstm/best.ckpt

# Transformer
python -m src.latency --config configs/power_transformer.yaml --ckpt experiments/power_transformer/best.ckpt
python -m src.latency --config configs/traffic_transformer.yaml --ckpt experiments/traffic_transformer/best.ckpt
python -m src.latency --config configs/weather_transformer.yaml --ckpt experiments/weather_transformer/best.ckpt

# Hybrid
python -m src.latency --config configs/power_hybrid.yaml --ckpt experiments/power_hybrid/best.ckpt
python -m src.latency --config configs/traffic_hybrid.yaml --ckpt experiments/traffic_hybrid/best.ckpt
python -m src.latency --config configs/weather_hybrid.yaml --ckpt experiments/weather_hybrid/best.ckpt

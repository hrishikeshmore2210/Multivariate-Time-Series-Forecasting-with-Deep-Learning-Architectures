# Multivariate Time Series Forecasting with LSTM, Transformer, and Hybrid Models

## 📌 Overview
This project implements and compares **three deep learning architectures** — **LSTM**, **Transformer**, and **Hybrid LSTM–Transformer** — for **multivariate time series forecasting** across three domains:
- **Electricity Load Forecasting** (Power Dataset)
- **Traffic Flow Prediction** (Traffic Dataset)
- **Weather Prediction** (Weather Dataset)

The models are evaluated on **accuracy**, **latency**, and **error patterns** to determine their suitability for real-world forecasting applications.

---

## 📂 Project Structure
```
.
├── configs/                  # YAML configuration files for each dataset/model
├── data/                     # Downloaded datasets & processed train/val/test splits
│   ├── power/
│   ├── traffic/
│   └── weather/
├── scripts/                  # Helper scripts for batch evaluation and analysis
├── src/                      # Model definitions and training loop
│   ├── models/               # LSTM, Transformer, Hybrid implementations
│   ├── train.py              # Training entry point
│   ├── model_registry.py     # Model factory
│   ├── utils.py              # Utility functions
├── references.bib            # Bibliography for the IEEE paper
├── final_full_report.tex     # Main IEEE-format report
├── README.md                 # This file
└── requirements.txt          # Python dependencies
```

---

## 📊 Datasets
1. **Power Dataset (UCI Electricity Load Diagrams)**  
   - Source: UCI Machine Learning Repository  
   - Hourly aggregated electricity consumption for multiple clients.
   
2. **Traffic Dataset (Monash Traffic)**  
   - Hourly road occupancy rates from multiple highway sensors.  
   
3. **Weather Dataset (Jena Climate)**  
   - Hourly temperature, pressure, humidity, wind speed, etc.

---

## 🛠 Preprocessing Pipeline
- Convert all columns to numeric
- Fill missing values with forward and backward fill
- Select target variable as the highest-variance series
- Select top features by variance
- Normalize using Z-score standardization
- Chronological split into **70% train**, **10% validation**, **20% test**

---

## 🧠 Model Architectures
### LSTM
Two stacked LSTM layers with dropout regularization, followed by a dense output layer.

### Transformer
Two encoder layers with multi-head self-attention and positional encoding.

### Hybrid LSTM–Transformer
An initial LSTM layer for local patterns followed by Transformer encoder layers for global dependencies.

---

## 📈 Evaluation Metrics
- **MAE** – Mean Absolute Error
- **RMSE** – Root Mean Squared Error
- **MAPE** – Mean Absolute Percentage Error
- **SMAPE** – Symmetric Mean Absolute Percentage Error
- **Inference Latency** – Average batch prediction time

---

## 🚀 Running the Project

### 1. Clone the Repository
```bash
git clone <repo_url>
cd <repo_name>
```

### 2. Create a Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate    # Linux/Mac
.venv\Scripts\activate       # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download & Prepare Datasets
Run preprocessing scripts for each dataset:
```bash
python scripts/prep_power.py
python scripts/prep_traffic.py
python scripts/prep_weather.py
```

### 5. Train Models
Example:
```bash
python -m src.train --config configs/power.yaml
```

### 6. Batch Evaluation
```bash
python scripts/run_all.py
```

---

## 📜 Paper
The complete IEEE-format paper for this project can be found in:
- `final_full_report_with_cites.tex`
- `references.bib`

---

## 📌 Notes
- Ensure datasets are placed in `data/` before training.
- Model checkpoints (`best.ckpt`) will be saved in `outputs/`.
- For reproducibility, all configs specify `random_seed`.

---

## 📄 License
This project is released under the MIT License.

from .models.lstm_baseline import LSTMBaseline
from .models.transformer_baseline import TransformerBaseline
from .models.hybrid_model import HybridLSTMTransformer

def create_model(name: str, **kwargs):
    name = name.lower()
    if name == "lstm":
        return LSTMBaseline(**kwargs)
    if name == "transformer":
        return TransformerBaseline(**kwargs)
    if name == "hybrid":
        return HybridLSTMTransformer(**kwargs)
    raise ValueError(f"Unknown model name: {name}")

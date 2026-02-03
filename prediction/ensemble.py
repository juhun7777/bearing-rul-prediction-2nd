"""
앙상블 예측 기법
"""

import numpy as np
import torch

def forecast_with_transformer(model, hi_part, full_len, src_len=3, safety_factor=0.15):
    """Transformer 모델을 사용한 예측 함수"""
    hi_pred = hi_part.tolist()
    for _ in range(full_len - len(hi_part)):
        inp = torch.tensor(hi_pred[-src_len:], dtype=torch.float32).unsqueeze(0)
        next_val = model(inp).item()
        next_val = next_val + safety_factor * (1.0 - next_val)
        hi_pred.append(next_val)
    return np.array(hi_pred, dtype=np.float32)

def ensemble_forecast(models, hi_part, full_len, src_len=3, early_bias=0.15):
    """앙상블 예측 함수"""
    forecasts = []
    for model in models:
        hi_pred = forecast_with_transformer(model, hi_part, full_len, src_len)
        forecasts.append(hi_pred)
    
    stacked = np.stack(forecasts)
    ensemble_pred = np.mean(stacked, axis=0) + early_bias * np.max(stacked, axis=0)
    ensemble_pred = np.clip(ensemble_pred, 0, 1)
    
    return ensemble_pred.astype(np.float32)

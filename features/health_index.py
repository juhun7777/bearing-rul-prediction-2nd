"""
Health Index 생성 및 PCA 관련 기능
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from config.settings import INTERVAL_MIN

def create_health_index(rms_data, channels):
    """PCA 기반 Health Index 생성 함수"""
    data_matrix = np.array([rms_data[ch] for ch in channels], dtype=np.float32).T
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data_matrix)
    pca = PCA(n_components=1)
    hi = pca.fit_transform(data_normalized).flatten()
    return hi, scaler, pca


def train_pca_model_from_data(train_data, train_ids, weights=None):
    """학습 데이터로부터 PCA 모델 생성 함수"""
    if weights is None:
        weights = [1.0] * len(train_data)
    
    normalized_data = []
    lifetimes = []
    raw_data = train_data
    
    for i, mat in enumerate(raw_data):
        repeats = max(1, int(weights[i] * 100))
        for _ in range(repeats):
            normalized_data.append(mat.astype(np.float32))
        
        lifetime = len(mat) * INTERVAL_MIN / 60
        lifetimes.append(lifetime)
    
    combined = np.vstack(normalized_data).astype(np.float32)
    
    pca = PCA(n_components=1)
    hi_values = pca.fit_transform(combined).flatten()
    
    if np.corrcoef(np.arange(len(hi_values)), hi_values)[0,1] < 0:
        hi_values = -hi_values
    
    scaler = MinMaxScaler((0,1))
    scaler.fit(hi_values.reshape(-1,1))
    
    # 시간 정규화 및 보간 과정
    ref_hi_normalized = []
    
    for i, mat in enumerate(raw_data):
        hi = pca.transform(mat.astype(np.float32)).flatten()
        if np.corrcoef(np.arange(len(hi)), hi)[0,1] < 0:
            hi = -hi
        hi_norm = scaler.transform(hi.reshape(-1,1)).flatten()
        time_norm = np.linspace(0, 1, len(hi_norm), dtype=np.float32)
        ref_hi_normalized.append((time_norm, hi_norm))
    
    time_points = np.linspace(0, 1, 100, dtype=np.float32)
    combined_hi = np.zeros_like(time_points, dtype=np.float32)
    
    for time_norm, hi_norm in ref_hi_normalized:
        interpolated_hi = np.interp(time_points, time_norm, hi_norm)
        combined_hi += interpolated_hi
    
    combined_hi /= len(ref_hi_normalized)
    
    avg_life = np.mean(lifetimes)
    
    return pca, scaler, avg_life, (time_points, combined_hi)

def calculate_hi_with_trained_pca(rms_data, pca, scaler, channels):
    """학습된 PCA로 HI 계산"""
    mat = np.column_stack([rms_data[ch] for ch in channels])
    hi = pca.transform(mat).flatten()
    if np.corrcoef(np.arange(len(hi)), hi)[0,1] < 0:
        hi = -hi
    return scaler.transform(hi.reshape(-1,1)).flatten()

def calculate_rul_from_hi(hi, total_life, safety_factor=0.85):
    """HI로부터 RUL 계산"""
    hi_clip = np.clip(hi, 0, 0.99)
    return ((1 - hi_clip) * total_life * safety_factor).astype(np.float32)

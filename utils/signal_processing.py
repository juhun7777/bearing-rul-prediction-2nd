"""
신호 처리 유틸리티 함수들
"""

import numpy as np
from scipy import signal
from config.settings import FS

def preprocess_signal(x, threshold=3.0):
    """신호 전처리 함수 - 이상치 제거 및 필터링 (float32 형식 준수)"""
    # float32 형식으로 변환 (제출 규정 준수)
    x = np.array(x, dtype=np.float32)
    
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    mask = np.abs(x - med) > threshold * mad
    clean = x.copy()
    if mask.any():
        clean[mask] = np.interp(
            np.flatnonzero(mask),
            np.flatnonzero(~mask),
            x[~mask]
        )
    b, a = signal.butter(4, 25/(FS/2), btype='highpass')
    filt = signal.filtfilt(b, a, clean)
    for f0 in (50, 60):
        w0 = f0/(FS/2)
        b, a = signal.iirnotch(w0, Q=30)
        filt = signal.filtfilt(b, a, filt)
    return filt.astype(np.float32)

def remove_spikes_from_rms(rms_vals, threshold=2.5):
    """RMS 값의 스파이크 노이즈 제거 함수 (float32 형식 준수)"""
    # float32 형식으로 변환 (제출 규정 준수)
    arr = np.array(rms_vals, dtype=np.float32)
    if arr.size < 3:
        return arr
    med = np.zeros_like(arr, dtype=np.float32)
    for i in range(arr.size):
        s, e = max(0, i-2), min(arr.size, i+3)
        med[i] = np.median(arr[s:e])
    dev = np.abs(arr - med)
    m = np.median(dev)
    mask = dev > threshold * m
    if mask.any():
        idx_out = np.flatnonzero(mask)
        idx_in  = np.flatnonzero(~mask)
        if idx_in.size == 0:
            return np.full_like(arr, np.median(arr), dtype=np.float32)
        arr[idx_out] = np.interp(idx_out, idx_in, arr[idx_in])
    return arr.astype(np.float32)

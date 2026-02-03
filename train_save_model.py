import os
import glob
import math
import numpy as np
from scipy import signal
from nptdms import TdmsFile
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn

# ===== 전역 설정 =====
INTERVAL_MIN = 10    # 10분 간격
DURATION_S = 10      # 10초 측정
VALIDATION_DIR = r"C:\세미나 공부\데이터 챌린지\Validation Set"
TRAIN_DIR = r"C:\세미나 공부\데이터 챌린지\Train"
CHANNELS = ['CH1','CH2','CH3','CH4'] 
FS = 25600.0  # 샘플링 주파수 (Hz)

TEST_ID = 4
TRAIN_IDS = [1,3]
CUTOFF_PERCENT = 50

# ===== 신호 처리 함수 =====
def preprocess_signal(x, threshold=3.0):
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
    return filt

def remove_spikes_from_rms(rms_vals, threshold=2.5):
    arr = np.array(rms_vals, dtype=float)
    if arr.size < 3:
        return arr
    med = np.zeros_like(arr)
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
            return np.full_like(arr, np.median(arr))
        arr[idx_out] = np.interp(idx_out, idx_in, arr[idx_in])
    return arr

# ===== 데이터 로드 함수 =====
def load_and_process_bearing(bearing_id, is_validation=False, full_load=True):
    if is_validation:
        folder = os.path.join(VALIDATION_DIR, f"Validation{bearing_id}")
    else:
        folder = os.path.join(TRAIN_DIR, f"Train{bearing_id}")
    
    files = sorted(glob.glob(os.path.join(folder, "*.tdms")))
    if not files:
        raise FileNotFoundError(f"No TDMS files in {folder}")
    
    if not is_validation and not full_load:
        cutoff = int(len(files) * CUTOFF_PERCENT / 100)
        files = files[:cutoff]
    
    channel_rms = {ch: [] for ch in CHANNELS}
    for fp in files:
        td = TdmsFile.read(fp)
        grp = td.groups()[0].name
        vib = {ch.name: ch.data for ch in td[grp].channels()}
        for ch in CHANNELS:
            if ch in vib:
                filt = preprocess_signal(vib[ch])
                channel_rms[ch].append(np.sqrt(np.mean(filt**2)))
    
    for ch in CHANNELS:
        channel_rms[ch] = remove_spikes_from_rms(channel_rms[ch])
    
    return channel_rms, len(files)

def load_all_bearing_data(test_id, train_ids):
    bearing_data = {}
    
    bearing_data['validation'] = {}
    bearing_data['validation']['full'], _ = load_and_process_bearing(test_id, is_validation=True, full_load=True)
    
    for bid in train_ids:
        bearing_data[bid] = {}
        bearing_data[bid]['full'], _ = load_and_process_bearing(bid, is_validation=False, full_load=True)
        bearing_data[bid]['part'], _ = load_and_process_bearing(bid, is_validation=False, full_load=False)
    
    return bearing_data

def analyze_bearing_similarity(test_id, candidate_ids, bearing_data):
    similarities = {}
    
    val_rms = bearing_data['validation']['full']
    val_duration_hours = len(val_rms[CHANNELS[0]]) * INTERVAL_MIN / 60
    val_mat = np.column_stack([val_rms[ch] for ch in CHANNELS])
    
    for bid in candidate_ids:
        train_rms = bearing_data[bid]['full']
        train_duration_hours = len(train_rms[CHANNELS[0]]) * INTERVAL_MIN / 60
        
        if train_duration_hours >= val_duration_hours:
            train_end_idx = int(len(train_rms[CHANNELS[0]]) * (val_duration_hours / train_duration_hours))
            train_mat = np.column_stack([train_rms[ch][:train_end_idx] for ch in CHANNELS])
        else:
            train_mat = np.column_stack([train_rms[ch] for ch in CHANNELS])
        
        min_len = min(len(val_mat), len(train_mat))
        val_slice = val_mat[:min_len]
        train_slice = train_mat[:min_len]
        
        correlation = np.mean([
            np.corrcoef(val_slice[:, i], train_slice[:, i])[0, 1]
            for i in range(len(CHANNELS))
        ])
        
        similarities[bid] = max(0, correlation)
    
    total = sum(similarities.values())
    if total > 0:
        weights = {bid: (sim/total * len(similarities)) for bid, sim in similarities.items()}
    else:
        weights = {bid: 1.0 for bid in similarities}
    
    return weights

# ===== PCA 기반 HI 생성 함수 =====
def train_pca_model_from_data(train_data, train_ids, weights=None):
    if weights is None:
        weights = [1.0] * len(train_data)
    
    normalized_data = []
    lifetimes = []
    raw_data = train_data
    
    for i, mat in enumerate(raw_data):
        repeats = max(1, int(weights[i] * 100))
        for _ in range(repeats):
            normalized_data.append(mat)
        
        lifetime = len(mat) * INTERVAL_MIN / 60
        lifetimes.append(lifetime)
    
    combined = np.vstack(normalized_data)
    
    pca = PCA(n_components=1)
    hi_values = pca.fit_transform(combined).flatten()
    
    if np.corrcoef(np.arange(len(hi_values)), hi_values)[0,1] < 0:
        hi_values = -hi_values
    
    scaler = MinMaxScaler((0,1))
    scaler.fit(hi_values.reshape(-1,1))
    
    ref_hi_normalized = []
    
    for i, mat in enumerate(raw_data):
        hi = pca.transform(mat).flatten()
        if np.corrcoef(np.arange(len(hi)), hi)[0,1] < 0:
            hi = -hi
        hi_norm = scaler.transform(hi.reshape(-1,1)).flatten()
        time_norm = np.linspace(0, 1, len(hi_norm))
        ref_hi_normalized.append((time_norm, hi_norm))
    
    time_points = np.linspace(0, 1, 100)
    combined_hi = np.zeros_like(time_points)
    
    for time_norm, hi_norm in ref_hi_normalized:
        interpolated_hi = np.interp(time_points, time_norm, hi_norm)
        combined_hi += interpolated_hi
    
    combined_hi /= len(ref_hi_normalized)
    
    avg_life = np.mean(lifetimes)
    
    return pca, scaler, avg_life, (time_points, combined_hi)

def calculate_hi_with_trained_pca(rms_data, pca, scaler):
    mat = np.column_stack([rms_data[ch] for ch in CHANNELS])
    hi = pca.transform(mat).flatten()
    if np.corrcoef(np.arange(len(hi)), hi)[0,1] < 0:
        hi = -hi
    return scaler.transform(hi.reshape(-1,1)).flatten()

def calculate_rul_from_hi(hi, total_life, safety_factor=0.85):
    hi_clip = np.clip(hi, 0, 0.99)
    return (1 - hi_clip) * total_life * safety_factor

# ===== Transformer 모델 =====
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0,max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos*div)
        pe[:,1::2] = torch.cos(pos*div)
        self.pe = pe.unsqueeze(1)
    def forward(self, x):
        return x + self.pe[:x.size(0)]

class TimeSeriesTransformer(nn.Module):
    def __init__(self, src_len, d_model=64, nhead=8, num_layers=3, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.input_emb = nn.Linear(2, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=src_len)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers)
        self.linear = nn.Linear(d_model, 1)
        
    def forward(self, src):
        x = src.unsqueeze(2)
        
        batch_size, seq_len = src.shape
        rates = torch.zeros((batch_size, seq_len, 1), device=src.device)
        rates[:, 1:, 0] = src[:, 1:] - src[:, :-1]
        
        x = torch.cat([x, rates], dim=2)
        x = self.input_emb(x)
        x = x.permute(1, 0, 2)
        x = self.pos_enc(x)
        out = self.transformer(x)
        last = out[-1, :, :]
        return self.linear(last).squeeze(1)

def forecast_with_transformer(model, hi_part, full_len, src_len=3, safety_factor=0.15):
    hi_pred = hi_part.tolist()
    for _ in range(full_len - len(hi_part)):
        inp = torch.tensor(hi_pred[-src_len:], dtype=torch.float32).unsqueeze(0)
        next_val = model(inp).item()
        next_val = next_val + safety_factor * (1.0 - next_val)
        hi_pred.append(next_val)
    return np.array(hi_pred)

def ensemble_forecast(models, hi_part, full_len, src_len=3, early_bias=0.15):
    forecasts = []
    for model in models:
        hi_pred = forecast_with_transformer(model, hi_part, full_len, src_len)
        forecasts.append(hi_pred)
    
    stacked = np.stack(forecasts)
    ensemble_pred = np.mean(stacked, axis=0) + early_bias * np.max(stacked, axis=0)
    ensemble_pred = np.clip(ensemble_pred, 0, 1)
    
    return ensemble_pred

# ===== 메인 함수 =====
def main():
    # 1. 데이터 로드
    bearing_data = load_all_bearing_data(TEST_ID, TRAIN_IDS)
    
    # 2. 유사도 분석
    train_weights = analyze_bearing_similarity(TEST_ID, TRAIN_IDS, bearing_data)
    
    # 3. PCA 모델 학습
    train_data = []
    train_weights_list = []
    for bid in TRAIN_IDS:
        mat = np.column_stack([bearing_data[bid]['full'][ch] for ch in CHANNELS])
        train_data.append(mat)
        train_weights_list.append(train_weights[bid])
    
    pca, scaler, avg_life, ref_hi_data = train_pca_model_from_data(
        train_data, TRAIN_IDS, weights=train_weights_list)
    
    # 4. Validation HI 계산
    val_hi = calculate_hi_with_trained_pca(bearing_data['validation']['full'], pca, scaler)
    
    # 5. 저장된 모델 로드
    models = []
    src_len = 3
    for i in range(1, 4):
        model = TimeSeriesTransformer(src_len)
        model_path = f"C:/세미나 공부/데이터 챌린지/본 파일/saved_models/Valid4_transformer_ensemble_{i}.pt"
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        models.append(model)
    
    # 6. 현재 상태 계산
    current_time = len(val_hi) * INTERVAL_MIN / 60
    current_hi = val_hi[-1]
    
    # 7. 미래 예측
    remaining_life_estimate = avg_life - current_time
    if remaining_life_estimate > 0:
        pred_hours = min(remaining_life_estimate * 1.5, avg_life)
    else:
        pred_hours = avg_life * 0.5
    
    pred_length = int(pred_hours * 60 / INTERVAL_MIN)
    seed_length = min(30, len(val_hi))
    val_hi_seed = val_hi[-seed_length:]
    
    if seed_length >= src_len and pred_length > 0:
        hi_pred_future = ensemble_forecast(
            models, val_hi_seed, seed_length + pred_length, 
            src_len=src_len, early_bias=0.15
        )
        hi_pred = hi_pred_future[seed_length:]
        
        if len(hi_pred) != pred_length:
            if len(hi_pred) > pred_length:
                hi_pred = hi_pred[:pred_length]
            elif len(hi_pred) < pred_length:
                last_val = hi_pred[-1] if len(hi_pred) > 0 else current_hi
                padding = np.full(pred_length - len(hi_pred), last_val)
                hi_pred = np.concatenate([hi_pred, padding])
    else:
        hi_pred = np.linspace(current_hi, min(current_hi + 0.3, 1.0), max(1, pred_length))
    
    # 8. RUL 계산
    safety_factor =0.8
    rul_pred = calculate_rul_from_hi(hi_pred, avg_life, safety_factor)
    
    # 9. 결과 출력
    print(f"===== Validation{TEST_ID} 베어링 RUL 예측 결과 =====")
    print(f"현재 운영 시간: {current_time:.2f}시간")
    if len(rul_pred) > 0:
        print(f"현재 시점 예측 RUL: {rul_pred[0]:.2f}시간")

if __name__ == "__main__":
    main()

    

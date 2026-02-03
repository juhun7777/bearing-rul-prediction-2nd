import os
import glob
import math
import numpy as np
from scipy import signal
from nptdms import TdmsFile
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ===== 전역 설정 =====
INTERVAL_MIN = 10    # 10분 간격
DURATION_S = 10      # 10초 측정
FS = 25600.0  # 샘플링 주파수 (Hz)

if os.path.exists("/data"):
    # 채점 서버 환경 (/data 디렉터리 존재)
    VALIDATION_DIR = "/data/Validation Set"
    TRAIN_DIR = "/data/Train"
    print("채점 서버 환경에서 실행 중...")
else:
    # 로컬 개발 환경
    VALIDATION_DIR = "Validation Set"
    TRAIN_DIR = "Train"
    print("로컬 환경에서 실행 중...")

# 모델 파일은 항상 제출 코드와 함께 위치
MODEL_DIR = "saved_models"

# TEST_ID 설정 (특정 번호, 리스트로 전체 실행)
TEST_ID = [1,2,3,4,5,6]  # 예측할 Validation 베어링 번호들

# TEST_ID에 따라 자동으로 설정 (기본값, 리스트일 때는 main에서 처리)
if not isinstance(TEST_ID, list):
    if TEST_ID in [1, 2, 3, 4]:
        # Validation 1,2,3,4용 설정
        CHANNELS = ['CH1','CH2','CH3','CH4']
        
        if TEST_ID in [2, 4]:
            TRAIN_IDS = [1, 3]
        elif TEST_ID in [1, 3]:
            TRAIN_IDS = [4, 5, 7]
            
        CUTOFF_PERCENT = 50
        MODEL_TYPE = "TimeSeriesTransformer"
        
    elif TEST_ID in [5, 6]:
        # Validation 5,6용 설정
        CHANNELS = ['CH1','CH2','CH3']
        TRAIN_IDS = [6]
        CUTOFF_PERCENT = 100
        MODEL_TYPE = "TransformerRULModel"
        
    else:
        raise ValueError(f"지원하지 않는 TEST_ID: {TEST_ID}. 1, 2, 3, 4, 5, 6 중 하나를 선택하세요.")
else:
    # 리스트일 때는 기본값으로 설정 (실제 값은 각 함수에서 동적으로 설정)
    CHANNELS = ['CH1','CH2','CH3','CH4']
    TRAIN_IDS = [1, 3]
    CUTOFF_PERCENT = 50
    MODEL_TYPE = "TimeSeriesTransformer"

# 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

# ===== 신호 처리 함수 =====
def preprocess_signal(x, threshold=3.0):
    """신호 전처리 함수 - 이상치 제거 및 필터링"""
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
    """RMS 값의 스파이크 노이즈 제거 함수"""
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

# ===== 데이터 로드 함수 =====
def load_and_process_bearing(bearing_id, is_validation=False, full_load=True):
    """베어링 데이터 로드 및 처리 함수"""
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
    for fp in tqdm(files, disable=True):
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
    """모든 베어링 데이터 로드 함수"""
    bearing_data = {}
    
    bearing_data['validation'] = {}
    bearing_data['validation']['full'], _ = load_and_process_bearing(test_id, is_validation=True, full_load=True)
    
    for bid in train_ids:
        bearing_data[bid] = {}
        bearing_data[bid]['full'], _ = load_and_process_bearing(bid, is_validation=False, full_load=True)
        bearing_data[bid]['part'], _ = load_and_process_bearing(bid, is_validation=False, full_load=False)
    
    return bearing_data

def analyze_bearing_similarity(test_id, candidate_ids, bearing_data):
    """베어링 유사도 분석 함수"""
    similarities = {}
    
    val_rms = bearing_data['validation']['full']
    val_duration_hours = len(val_rms[CHANNELS[0]]) * INTERVAL_MIN / 60
    val_mat = np.column_stack([val_rms[ch] for ch in CHANNELS]).astype(np.float32)
    
    for bid in candidate_ids:
        train_rms = bearing_data[bid]['full']
        train_duration_hours = len(train_rms[CHANNELS[0]]) * INTERVAL_MIN / 60
        
        if train_duration_hours >= val_duration_hours:
            train_end_idx = int(len(train_rms[CHANNELS[0]]) * (val_duration_hours / train_duration_hours))
            train_mat = np.column_stack([train_rms[ch][:train_end_idx] for ch in CHANNELS]).astype(np.float32)
        else:
            train_mat = np.column_stack([train_rms[ch] for ch in CHANNELS]).astype(np.float32)
        
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

# ===== Health Index 생성 함수 =====
def create_health_index(rms_data):
    """PCA 기반 Health Index 생성 함수"""
    # float32 형식으로 변환 (제출 규정 준수)
    data_matrix = np.array([rms_data[ch] for ch in CHANNELS], dtype=np.float32).T
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

def calculate_hi_with_trained_pca(rms_data, pca, scaler):
    """학습된 PCA로 Health Index 계산 함수"""
    mat = np.column_stack([rms_data[ch] for ch in CHANNELS]).astype(np.float32)
    hi = pca.transform(mat).flatten()
    if np.corrcoef(np.arange(len(hi)), hi)[0,1] < 0:
        hi = -hi
    return scaler.transform(hi.reshape(-1,1)).flatten()

# ===== 시각화 함수 =====
def visualize_pca_hi_process(rms_data, pca, scaler):
    """PCA 기반 HI 생성 과정 시각화"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('PCA-based Health Index Generation Process', fontsize=16, fontweight='bold')
    
    # Step 1: 4채널 RMS 입력 데이터
    ax1 = axes[0, 0]
    colors = ['blue', 'red', 'green', 'orange']
    for i, ch in enumerate(CHANNELS):
        ax1.plot(rms_data[ch], color=colors[i], label=f'{ch}', linewidth=1.5)
    ax1.set_title('Step 1: 4-Channel RMS Input', fontweight='bold')
    ax1.set_xlabel('Time Points')
    ax1.set_ylabel('RMS Value')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Step 2: PCA 변환 결과 (HI)
    ax2 = axes[0, 1]
    mat = np.column_stack([rms_data[ch] for ch in CHANNELS]).astype(np.float32)
    hi_raw = pca.transform(mat).flatten()
    if np.corrcoef(np.arange(len(hi_raw)), hi_raw)[0,1] < 0:
        hi_raw = -hi_raw
    ax2.plot(hi_raw, 'purple', linewidth=2, label='Raw HI')
    ax2.set_title('Step 2: PCA Transformation (4D→1D)', fontweight='bold')
    ax2.set_xlabel('Time Points')
    ax2.set_ylabel('HI Value')
    ax2.grid(True, alpha=0.3)
    
    # Step 3: 정규화된 최종 HI
    ax3 = axes[0, 2]
    hi_final = scaler.transform(hi_raw.reshape(-1,1)).flatten()
    ax3.plot(hi_final, 'red', linewidth=2, label='Normalized HI')
    ax3.axhline(y=0.8, color='orange', linestyle='--', alpha=0.7, label='Warning Level')
    ax3.set_title('Step 3: Normalized HI [0,1]', fontweight='bold')
    ax3.set_xlabel('Time Points')
    ax3.set_ylabel('Health Index')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # PCA 주성분 로딩 (채널별 기여도)
    ax4 = axes[1, 0]
    loadings = pca.components_[0]
    bars = ax4.bar(CHANNELS, loadings, color=['blue', 'red', 'green', 'orange'], alpha=0.7)
    ax4.set_title('PCA Component Loadings', fontweight='bold')
    ax4.set_xlabel('Channels')
    ax4.set_ylabel('Loading Value')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 로딩 값 표시
    for bar, loading in zip(bars, loadings):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01*np.sign(height),
                f'{loading:.3f}', ha='center', va='bottom' if height > 0 else 'top')
    
    # 4채널 vs 1채널 비교
    ax5 = axes[1, 1]
    # 4채널 데이터의 표준편차 (복잡도 지표)
    std_4ch = [np.std(rms_data[ch]) for ch in CHANNELS]
    std_1ch = np.std(hi_final)
    
    x_pos = np.arange(len(CHANNELS) + 1)
    all_stds = std_4ch + [std_1ch]
    colors_extended = colors + ['red']
    labels_extended = CHANNELS + ['HI']
    
    bars = ax5.bar(x_pos, all_stds, color=colors_extended, alpha=0.7)
    ax5.set_title('Data Complexity: 4CH → 1CH', fontweight='bold')
    ax5.set_xlabel('Channels')
    ax5.set_ylabel('Standard Deviation')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(labels_extended)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 차원 축소 효과 표시
    ax6 = axes[1, 2]
    explained_var = pca.explained_variance_ratio_[0]
    categories = ['Original\n(4 Channels)', 'Compressed\n(1 Channel)']
    info_preserved = [1.0, explained_var]
    
    bars = ax6.bar(categories, info_preserved, color=['lightblue', 'red'], alpha=0.7)
    ax6.set_title('Information Preservation', fontweight='bold')
    ax6.set_ylabel('Information Ratio')
    ax6.set_ylim(0, 1.1)
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 정보 보존율 표시
    for bar, ratio in zip(bars, info_preserved):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{ratio:.1%}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    return fig

def visualize_signal_processing_comparison(raw_signal):
    """신호처리 전후 비교 시각화"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Multi-stage Noise Removal and Feature Extraction Process', fontsize=16, fontweight='bold')
    
    # 원시 신호
    ax1 = axes[0, 0]
    time_raw = np.arange(len(raw_signal)) / FS
    ax1.plot(time_raw[:5000], raw_signal[:5000], 'blue', linewidth=0.5)
    ax1.set_title('1. Raw Signal', fontweight='bold')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)
    
    # 이상치 제거 후
    med = np.median(raw_signal)
    mad = np.median(np.abs(raw_signal - med))
    mask = np.abs(raw_signal - med) > 3.0 * mad
    clean_signal = raw_signal.copy()
    if mask.any():
        clean_signal[mask] = np.interp(
            np.flatnonzero(mask),
            np.flatnonzero(~mask),
            raw_signal[~mask]
        )
    
    ax2 = axes[0, 1]
    ax2.plot(time_raw[:5000], clean_signal[:5000], 'green', linewidth=0.5)
    if mask.any():
        outlier_indices = np.flatnonzero(mask)[:50]  # 처음 50개만 표시
        ax2.scatter(time_raw[outlier_indices], raw_signal[outlier_indices], 
                   color='red', s=10, alpha=0.7, label=f'Outliers ({np.sum(mask)})')
        ax2.legend(fontsize=8)
    ax2.set_title('2. Outlier Removal (MAD)', fontweight='bold')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True, alpha=0.3)
    
    # 고역통과 필터링 후
    b, a = signal.butter(4, 25/(FS/2), btype='highpass')
    filtered_signal = signal.filtfilt(b, a, clean_signal)
    
    ax3 = axes[0, 2]
    ax3.plot(time_raw[:5000], filtered_signal[:5000], 'orange', linewidth=0.5)
    ax3.set_title('3. High-pass Filter (25Hz)', fontweight='bold')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Amplitude')
    ax3.grid(True, alpha=0.3)
    
    # 노치 필터링 후
    final_signal = filtered_signal.copy()
    for f0 in (50, 60):
        w0 = f0/(FS/2)
        b, a = signal.iirnotch(w0, Q=30)
        final_signal = signal.filtfilt(b, a, final_signal)
    
    ax4 = axes[1, 0]
    ax4.plot(time_raw[:5000], final_signal[:5000], 'red', linewidth=0.5)
    ax4.set_title('4. Notch Filter (50/60Hz)', fontweight='bold')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Amplitude')
    ax4.grid(True, alpha=0.3)
    
    # 주파수 도메인 비교
    ax5 = axes[1, 1]
    freqs_raw, psd_raw = signal.welch(raw_signal, FS, nperseg=8192)
    freqs_final, psd_final = signal.welch(final_signal, FS, nperseg=8192)
    
    ax5.loglog(freqs_raw, psd_raw, 'blue', alpha=0.7, label='Raw Signal')
    ax5.loglog(freqs_final, psd_final, 'red', alpha=0.7, label='Processed Signal')
    ax5.axvline(x=25, color='orange', linestyle='--', alpha=0.7, label='25Hz Cutoff')
    ax5.axvline(x=50, color='green', linestyle='--', alpha=0.7, label='50Hz Notch')
    ax5.axvline(x=60, color='green', linestyle='--', alpha=0.7, label='60Hz Notch')
    ax5.set_title('5. Frequency Domain Comparison', fontweight='bold')
    ax5.set_xlabel('Frequency (Hz)')
    ax5.set_ylabel('PSD')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # RMS 추출 비교
    ax6 = axes[1, 2]
    
    # 윈도우별 RMS 계산 (10초 윈도우)
    window_size = int(FS * 10)  # 10초
    num_windows = len(raw_signal) // window_size
    
    rms_raw = []
    rms_processed = []
    
    for i in range(num_windows):
        start_idx = i * window_size
        end_idx = start_idx + window_size
        
        rms_raw.append(np.sqrt(np.mean(raw_signal[start_idx:end_idx]**2)))
        rms_processed.append(np.sqrt(np.mean(final_signal[start_idx:end_idx]**2)))
    
    time_windows = np.arange(num_windows) * 10 / 60  # 분 단위
    
    ax6.plot(time_windows, rms_raw, 'blue', linewidth=2, label='Raw RMS')
    ax6.plot(time_windows, rms_processed, 'red', linewidth=2, label='Processed RMS')
    ax6.set_title('6. RMS Feature Extraction', fontweight='bold')
    ax6.set_xlabel('Time (minutes)')
    ax6.set_ylabel('RMS Value')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def main():
    """메인 실행 함수"""
    print("PCA 기반 HI 생성 및 신호처리 시각화 시작...")
    
    try:
        # 베어링 데이터 로드 (Train1 사용)
        print("베어링 데이터 로딩 중...")
        rms_data, _ = load_and_process_bearing(1, is_validation=False, full_load=True)
        
        # PCA 모델 생성
        print("PCA 모델 생성 중...")
        train_data = [np.column_stack([rms_data[ch] for ch in CHANNELS]).astype(np.float32)]
        pca, scaler, avg_life, ref_curve = train_pca_model_from_data(train_data, [1])
        
        # PCA HI 생성 과정 시각화
        print("PCA HI 생성 과정 시각화 중...")
        fig1 = visualize_pca_hi_process(rms_data, pca, scaler)
        plt.show()
        
        # 원시 신호 로드를 위한 샘플 TDMS 파일 읽기
        print("원시 신호 데이터 로딩 중...")
        folder = os.path.join(TRAIN_DIR, "Train1")
        files = sorted(glob.glob(os.path.join(folder, "*.tdms")))
        
        if files:
            # 첫 번째 파일에서 원시 신호 추출
            td = TdmsFile.read(files[0])
            grp = td.groups()[0].name
            vib = {ch.name: ch.data for ch in td[grp].channels()}
            
            # CH1 신호 사용 (다른 채널도 가능)
            raw_signal = vib['CH1']
            
            # 신호처리 전후 비교 시각화
            print("신호처리 전후 비교 시각화 중...")
            fig2 = visualize_signal_processing_comparison(raw_signal)
            plt.show()
        else:
            print("TDMS 파일을 찾을 수 없습니다.")
        
        print("시각화 완료!")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        print("데모 데이터로 시각화를 수행합니다...")
        
        # 데모 데이터 생성
        np.random.seed(42)
        time_points = np.arange(100)
        
        # 가상의 RMS 데이터 생성 (베어링 열화 패턴 시뮬레이션)
        demo_rms_data = {}
        for i, ch in enumerate(CHANNELS):
            base_trend = 0.1 + 0.3 * (time_points / 100) ** 2  # 2차 증가
            noise = 0.02 * np.random.normal(0, 1, len(time_points))
            channel_offset = 0.01 * i  # 채널별 오프셋
            demo_rms_data[ch] = base_trend + noise + channel_offset
        
        # 데모용 PCA 모델 생성
        demo_matrix = np.column_stack([demo_rms_data[ch] for ch in CHANNELS]).astype(np.float32)
        demo_pca = PCA(n_components=1)
        demo_pca.fit(demo_matrix)
        demo_scaler = MinMaxScaler()
        demo_hi = demo_pca.transform(demo_matrix).flatten()
        demo_scaler.fit(demo_hi.reshape(-1, 1))
        
        # PCA 시각화
        fig1 = visualize_pca_hi_process(demo_rms_data, demo_pca, demo_scaler)
        plt.show()
        
        # 데모용 원시 신호 생성
        demo_raw_signal = np.random.normal(0, 1, 256000)  # 10초 신호
        fig2 = visualize_signal_processing_comparison(demo_raw_signal)
        plt.show()

if __name__ == "__main__":
    main()
    
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import matplotlib.font_manager as fm

# 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

def visualize_pca_hi_process(rms_data, pca, scaler):
    """PCA 기반 HI 생성 과정 시각화"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('PCA-based Health Index Generation Process', fontsize=16, fontweight='bold')
    
    # Step 1: 4채널 RMS 입력 데이터
    ax1 = axes[0, 0]
    colors = ['blue', 'red', 'green', 'orange']
    for i, ch in enumerate(CHANNELS):
        ax1.plot(rms_data[ch], color=colors[i], label=f'{ch}', linewidth=1.5)
    ax1.set_title('Step 1: 4-Channel RMS Input', fontweight='bold')
    ax1.set_xlabel('Time Points')
    ax1.set_ylabel('RMS Value')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Step 2: PCA 변환 결과 (HI)
    ax2 = axes[0, 1]
    mat = np.column_stack([rms_data[ch] for ch in CHANNELS]).astype(np.float32)
    hi_raw = pca.transform(mat).flatten()
    if np.corrcoef(np.arange(len(hi_raw)), hi_raw)[0,1] < 0:
        hi_raw = -hi_raw
    ax2.plot(hi_raw, 'purple', linewidth=2, label='Raw HI')
    ax2.set_title('Step 2: PCA Transformation (4D→1D)', fontweight='bold')
    ax2.set_xlabel('Time Points')
    ax2.set_ylabel('HI Value')
    ax2.grid(True, alpha=0.3)
    
    # Step 3: 정규화된 최종 HI
    ax3 = axes[0, 2]
    hi_final = scaler.transform(hi_raw.reshape(-1,1)).flatten()
    ax3.plot(hi_final, 'red', linewidth=2, label='Normalized HI')
    ax3.axhline(y=0.8, color='orange', linestyle='--', alpha=0.7, label='Warning Level')
    ax3.set_title('Step 3: Normalized HI [0,1]', fontweight='bold')
    ax3.set_xlabel('Time Points')
    ax3.set_ylabel('Health Index')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # PCA 주성분 로딩 (채널별 기여도)
    ax4 = axes[1, 0]
    loadings = pca.components_[0]
    bars = ax4.bar(CHANNELS, loadings, color=['blue', 'red', 'green', 'orange'], alpha=0.7)
    ax4.set_title('PCA Component Loadings', fontweight='bold')
    ax4.set_xlabel('Channels')
    ax4.set_ylabel('Loading Value')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 로딩 값 표시
    for bar, loading in zip(bars, loadings):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01*np.sign(height),
                f'{loading:.3f}', ha='center', va='bottom' if height > 0 else 'top')
    
    # 4채널 vs 1채널 비교
    ax5 = axes[1, 1]
    # 4채널 데이터의 표준편차 (복잡도 지표)
    std_4ch = [np.std(rms_data[ch]) for ch in CHANNELS]
    std_1ch = np.std(hi_final)
    
    x_pos = np.arange(len(CHANNELS) + 1)
    all_stds = std_4ch + [std_1ch]
    colors_extended = colors + ['red']
    labels_extended = CHANNELS + ['HI']
    
    bars = ax5.bar(x_pos, all_stds, color=colors_extended, alpha=0.7)
    ax5.set_title('Data Complexity: 4CH → 1CH', fontweight='bold')
    ax5.set_xlabel('Channels')
    ax5.set_ylabel('Standard Deviation')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(labels_extended)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 차원 축소 효과 표시
    ax6 = axes[1, 2]
    explained_var = pca.explained_variance_ratio_[0]
    categories = ['Original\n(4 Channels)', 'Compressed\n(1 Channel)']
    info_preserved = [1.0, explained_var]
    
    bars = ax6.bar(categories, info_preserved, color=['lightblue', 'red'], alpha=0.7)
    ax6.set_title('Information Preservation', fontweight='bold')
    ax6.set_ylabel('Information Ratio')
    ax6.set_ylim(0, 1.1)
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 정보 보존율 표시
    for bar, ratio in zip(bars, info_preserved):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{ratio:.1%}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    return fig

def visualize_signal_processing_comparison(raw_signal):
    """신호처리 전후 비교 시각화"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Multi-stage Noise Removal and Feature Extraction Process', fontsize=16, fontweight='bold')
    
    # 원시 신호
    ax1 = axes[0, 0]
    time_raw = np.arange(len(raw_signal)) / FS
    ax1.plot(time_raw[:5000], raw_signal[:5000], 'blue', linewidth=0.5)
    ax1.set_title('1. Raw Signal', fontweight='bold')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)
    
    # 이상치 제거 후
    med = np.median(raw_signal)
    mad = np.median(np.abs(raw_signal - med))
    mask = np.abs(raw_signal - med) > 3.0 * mad
    clean_signal = raw_signal.copy()
    if mask.any():
        clean_signal[mask] = np.interp(
            np.flatnonzero(mask),
            np.flatnonzero(~mask),
            raw_signal[~mask]
        )
    
    ax2 = axes[0, 1]
    ax2.plot(time_raw[:5000], clean_signal[:5000], 'green', linewidth=0.5)
    if mask.any():
        outlier_indices = np.flatnonzero(mask)[:50]  # 처음 50개만 표시
        ax2.scatter(time_raw[outlier_indices], raw_signal[outlier_indices], 
                   color='red', s=10, alpha=0.7, label=f'Outliers ({np.sum(mask)})')
        ax2.legend(fontsize=8)
    ax2.set_title('2. Outlier Removal (MAD)', fontweight='bold')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True, alpha=0.3)
    
    # 고역통과 필터링 후
    b, a = signal.butter(4, 25/(FS/2), btype='highpass')
    filtered_signal = signal.filtfilt(b, a, clean_signal)
    
    ax3 = axes[0, 2]
    ax3.plot(time_raw[:5000], filtered_signal[:5000], 'orange', linewidth=0.5)
    ax3.set_title('3. High-pass Filter (25Hz)', fontweight='bold')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Amplitude')
    ax3.grid(True, alpha=0.3)
    
    # 노치 필터링 후
    final_signal = filtered_signal.copy()
    for f0 in (50, 60):
        w0 = f0/(FS/2)
        b, a = signal.iirnotch(w0, Q=30)
        final_signal = signal.filtfilt(b, a, final_signal)
    
    ax4 = axes[1, 0]
    ax4.plot(time_raw[:5000], final_signal[:5000], 'red', linewidth=0.5)
    ax4.set_title('4. Notch Filter (50/60Hz)', fontweight='bold')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Amplitude')
    ax4.grid(True, alpha=0.3)
    
    # 주파수 도메인 비교
    ax5 = axes[1, 1]
    freqs_raw, psd_raw = signal.welch(raw_signal, FS, nperseg=8192)
    freqs_final, psd_final = signal.welch(final_signal, FS, nperseg=8192)
    
    ax5.loglog(freqs_raw, psd_raw, 'blue', alpha=0.7, label='Raw Signal')
    ax5.loglog(freqs_final, psd_final, 'red', alpha=0.7, label='Processed Signal')
    ax5.axvline(x=25, color='orange', linestyle='--', alpha=0.7, label='25Hz Cutoff')
    ax5.axvline(x=50, color='green', linestyle='--', alpha=0.7, label='50Hz Notch')
    ax5.axvline(x=60, color='green', linestyle='--', alpha=0.7, label='60Hz Notch')
    ax5.set_title('5. Frequency Domain Comparison', fontweight='bold')
    ax5.set_xlabel('Frequency (Hz)')
    ax5.set_ylabel('PSD')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # RMS 추출 비교
    ax6 = axes[1, 2]
    
    # 윈도우별 RMS 계산 (10초 윈도우)
    window_size = int(FS * 10)  # 10초
    num_windows = len(raw_signal) // window_size
    
    rms_raw = []
    rms_processed = []
    
    for i in range(num_windows):
        start_idx = i * window_size
        end_idx = start_idx + window_size
        
        rms_raw.append(np.sqrt(np.mean(raw_signal[start_idx:end_idx]**2)))
        rms_processed.append(np.sqrt(np.mean(final_signal[start_idx:end_idx]**2)))
    
    time_windows = np.arange(num_windows) * 10 / 60  # 분 단위
    
    ax6.plot(time_windows, rms_raw, 'blue', linewidth=2, label='Raw RMS')
    ax6.plot(time_windows, rms_processed, 'red', linewidth=2, label='Processed RMS')
    ax6.set_title('6. RMS Feature Extraction', fontweight='bold')
    ax6.set_xlabel('Time (minutes)')
    ax6.set_ylabel('RMS Value')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# 사용 예시 (실제 데이터 로드 후 사용)

# 베어링 데이터 로드
rms_data, _ = load_and_process_bearing(1, is_validation=False, full_load=True)

# PCA 모델 생성
train_data = [np.column_stack([rms_data[ch] for ch in CHANNELS]).astype(np.float32)]
pca, scaler, avg_life, ref_curve = train_pca_model_from_data(train_data, [1])

# PCA HI 생성 과정 시각화
fig1 = visualize_pca_hi_process(rms_data, pca, scaler)
plt.show()

# 원시 신호 로드 (TDMS 파일에서)
# raw_signal = ... (실제 진동 신호 데이터)
# fig2 = visualize_signal_processing_comparison(raw_signal)
# plt.show()


import os
import glob
import numpy as np
from scipy import signal
from nptdms import TdmsFile
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ===== 전역 설정 =====
INTERVAL_MIN = 10    # 10분 간격
DURATION_S = 10      # 10초 측정
FS = 25600.0  # 샘플링 주파수 (Hz)
CHANNELS = ['CH1','CH2','CH3','CH4']

if os.path.exists("/data"):
    TRAIN_DIR = "/data/Train"
    print("채점 서버 환경에서 실행 중...")
else:
    TRAIN_DIR = "Train"
    print("로컬 환경에서 실행 중...")

# 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

def preprocess_signal_step_by_step(x, threshold=3.0):
    """단계별 신호 전처리 - 각 단계의 결과를 반환"""
    x = np.array(x, dtype=np.float32)
    
    # Step 1: 이상치 제거
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
    
    # Step 2: 고역통과 필터
    b, a = signal.butter(4, 25/(FS/2), btype='highpass')
    highpass_filtered = signal.filtfilt(b, a, clean)
    
    # Step 3: 노치 필터 (50Hz, 60Hz)
    notch_filtered = highpass_filtered.copy()
    for f0 in (50, 60):
        w0 = f0/(FS/2)
        b, a = signal.iirnotch(w0, Q=30)
        notch_filtered = signal.filtfilt(b, a, notch_filtered)
    
    return {
        'original': x.astype(np.float32),
        'outlier_removed': clean.astype(np.float32),
        'highpass_filtered': highpass_filtered.astype(np.float32),
        'notch_filtered': notch_filtered.astype(np.float32),
        'outlier_mask': mask
    }

def remove_spikes_from_rms(rms_vals, threshold=2.5):
    """RMS 값의 스파이크 노이즈 제거 함수"""
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

def process_entire_bearing_lifecycle(bearing_id=1):
    """전체 베어링 수명에 걸친 신호처리 과정 분석"""
    folder = os.path.join(TRAIN_DIR, f"Train{bearing_id}")
    files = sorted(glob.glob(os.path.join(folder, "*.tdms")))
    
    if not files:
        raise FileNotFoundError(f"No TDMS files in {folder}")
    
    print(f"Processing {len(files)} files for Train{bearing_id}...")
    
    # 각 단계별 RMS 값 저장
    rms_original = []
    rms_outlier_removed = []
    rms_final_processed = []
    time_points = []
    
    for i, fp in enumerate(tqdm(files, desc="Processing files")):
        try:
            td = TdmsFile.read(fp)
            grp = td.groups()[0].name
            vib = {ch.name: ch.data for ch in td[grp].channels()}
            
            # CH1 채널 사용 (대표 채널)
            if 'CH1' in vib:
                raw_signal = vib['CH1']
                
                # 단계별 처리
                processed = preprocess_signal_step_by_step(raw_signal)
                
                # 각 단계의 RMS 계산
                rms_original.append(np.sqrt(np.mean(processed['original']**2)))
                rms_outlier_removed.append(np.sqrt(np.mean(processed['outlier_removed']**2)))
                rms_final_processed.append(np.sqrt(np.mean(processed['notch_filtered']**2)))
                
                # 시간 정보
                time_points.append(i * INTERVAL_MIN / 60)  # 시간(시간 단위)
                
        except Exception as e:
            print(f"Error processing {fp}: {e}")
            continue
    
    # 스파이크 제거 적용
    rms_final_cleaned = remove_spikes_from_rms(rms_final_processed)
    
    return {
        'time_hours': np.array(time_points),
        'rms_original': np.array(rms_original),
        'rms_outlier_removed': np.array(rms_outlier_removed), 
        'rms_final_processed': np.array(rms_final_processed),
        'rms_final_cleaned': rms_final_cleaned,
        'total_files': len(files)
    }

def visualize_complete_signal_processing(data):
    """완전한 신호처리 과정 시각화 (전체 수명 기간)"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Complete Signal Processing Pipeline - Full Bearing Lifecycle', 
                 fontsize=16, fontweight='bold')
    
    time_hours = data['time_hours']
    
    # 1. 원신호 vs 이상치 제거
    ax1 = axes[0, 0]
    ax1.plot(time_hours, data['rms_original'], 'blue', linewidth=1.5, 
             label='Original Signal', alpha=0.8)
    ax1.plot(time_hours, data['rms_outlier_removed'], 'green', linewidth=1.5, 
             label='Outlier Removed', alpha=0.8)
    ax1.set_title('Step 1: Outlier Removal (MAD-based)', fontweight='bold')
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('RMS Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 필터링 과정 (고역통과 + 노치)
    ax2 = axes[0, 1]
    ax2.plot(time_hours, data['rms_outlier_removed'], 'green', linewidth=1.5, 
             label='After Outlier Removal', alpha=0.8)
    ax2.plot(time_hours, data['rms_final_processed'], 'orange', linewidth=1.5, 
             label='After HP+Notch Filter', alpha=0.8)
    ax2.set_title('Step 2: Filtering (25Hz HP + 50/60Hz Notch)', fontweight='bold')
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('RMS Value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 최종 스파이크 제거
    ax3 = axes[1, 0]
    ax3.plot(time_hours, data['rms_final_processed'], 'orange', linewidth=1.5, 
             label='Before Spike Removal', alpha=0.8)
    ax3.plot(time_hours, data['rms_final_cleaned'], 'red', linewidth=2, 
             label='Final Cleaned Signal', alpha=0.9)
    ax3.set_title('Step 3: RMS Spike Removal', fontweight='bold')
    ax3.set_xlabel('Time (hours)')
    ax3.set_ylabel('RMS Value')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 전체 과정 비교
    ax4 = axes[1, 1]
    ax4.plot(time_hours, data['rms_original'], 'blue', linewidth=1, 
             label='Original', alpha=0.6)
    ax4.plot(time_hours, data['rms_final_cleaned'], 'red', linewidth=2, 
             label='Final Processed', alpha=0.9)
    ax4.set_title('Complete Process: Original → Final', fontweight='bold')
    ax4.set_xlabel('Time (hours)')
    ax4.set_ylabel('RMS Value')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 개선 효과 텍스트 정보 추가
    improvement_text = f"""
Processing Results:
• Total Duration: {time_hours[-1]:.1f} hours
• Data Points: {len(time_hours)} measurements
• Noise Reduction: {(1 - np.std(data['rms_final_cleaned'])/np.std(data['rms_original']))*100:.1f}%
• Signal Smoothness: {np.std(np.diff(data['rms_final_cleaned']))/np.std(np.diff(data['rms_original'])):.2f}x better
    """
    
    fig.text(0.02, 0.02, improvement_text, fontsize=9, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    return fig

def visualize_processing_stages_comparison():
    """단계별 처리 효과 비교 시각화"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Signal Processing Stages - Flow Diagram Style', 
                 fontsize=16, fontweight='bold')
    
    # 샘플 데이터로 각 단계 효과 시뮬레이션
    t = np.linspace(0, 100, 1000)
    
    # 원신호 (트렌드 + 노이즈 + 이상치 + 주기성분)
    trend = 0.5 + 0.3 * (t/100)**2
    noise = 0.05 * np.random.normal(0, 1, len(t))
    periodic_50hz = 0.02 * np.sin(2*np.pi*50*t/100)  # 50Hz 성분
    periodic_60hz = 0.015 * np.sin(2*np.pi*60*t/100)  # 60Hz 성분
    
    # 이상치 추가
    outlier_idx = np.random.choice(len(t), size=20, replace=False)
    outliers = np.zeros_like(t)
    outliers[outlier_idx] = np.random.uniform(-0.3, 0.3, len(outlier_idx))
    
    original = trend + noise + periodic_50hz + periodic_60hz + outliers
    
    # Stage 1: 이상치 제거
    ax1 = axes[0]
    processed_stage1 = trend + noise + periodic_50hz + periodic_60hz
    
    ax1.plot(t, original, 'blue', alpha=0.7, linewidth=1, label='원신호')
    ax1.plot(t, processed_stage1, 'green', linewidth=2, label='이상치 제거 후')
    ax1.scatter(t[outlier_idx], original[outlier_idx], color='red', s=20, 
               alpha=0.8, label='제거된 이상치', zorder=5)
    ax1.set_title('1단계: 이상치 제거\n(MAD 기반 Robust 처리)', fontweight='bold')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Amplitude')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Stage 2: 필터링
    ax2 = axes[1]
    processed_stage2 = trend + 0.7*noise  # 고주파 노이즈 감소 + 전력선 주파수 제거
    
    ax2.plot(t, processed_stage1, 'green', alpha=0.7, linewidth=1, label='이상치 제거 후')
    ax2.plot(t, processed_stage2, 'orange', linewidth=2, label='필터링 후')
    ax2.set_title('2단계: 신호 필터링\n(25Hz 고역통과 + 50/60Hz 노치)', fontweight='bold')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Amplitude')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Stage 3: RMS 스파이크 제거
    ax3 = axes[2]
    # RMS 계산 시뮬레이션
    window_size = 50
    rms_with_spikes = []
    rms_cleaned = []
    
    for i in range(0, len(processed_stage2)-window_size, window_size//2):
        window = processed_stage2[i:i+window_size]
        rms_val = np.sqrt(np.mean(window**2))
        
        # 가끔 스파이크 추가
        if np.random.random() < 0.1:
            rms_val *= (1 + np.random.uniform(0.5, 1.0))
        
        rms_with_spikes.append(rms_val)
    
    # 스파이크 제거
    rms_cleaned = remove_spikes_from_rms(rms_with_spikes)
    
    t_rms = np.linspace(0, 100, len(rms_with_spikes))
    
    ax3.plot(t_rms, rms_with_spikes, 'orange', alpha=0.7, linewidth=1, 
             label='RMS (스파이크 포함)')
    ax3.plot(t_rms, rms_cleaned, 'red', linewidth=2, label='최종 정제된 RMS')
    ax3.set_title('3단계: RMS 추출 및 스파이크 제거\n(에너지 레벨 정량화)', fontweight='bold')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('RMS Value')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def main():
    """메인 실행 함수"""
    print("=== 개선된 신호처리 시각화 시작 ===")
    
    try:
        # 전체 베어링 수명 데이터 처리
        print("전체 베어링 수명 데이터 처리 중...")
        data = process_entire_bearing_lifecycle(bearing_id=1)
        
        # 완전한 신호처리 과정 시각화
        print("신호처리 과정 시각화 생성 중...")
        fig1 = visualize_complete_signal_processing(data)
        plt.show()
        
        # 단계별 처리 효과 비교
        print("단계별 처리 효과 시각화 생성 중...")
        fig2 = visualize_processing_stages_comparison()
        plt.show()
        
        print("시각화 완료!")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        print("데모 시각화를 실행합니다...")
        
        # 데모 데이터로 시각화
        demo_data = {
            'time_hours': np.linspace(0, 100, 500),
            'rms_original': None,
            'rms_outlier_removed': None,
            'rms_final_processed': None,
            'rms_final_cleaned': None,
            'total_files': 500
        }
        
        # 가상의 베어링 열화 데이터 생성
        t = demo_data['time_hours']
        base_trend = 0.1 + 0.4 * (t/100)**2  # 2차 증가 트렌드
        
        # 각 단계별 노이즈 레벨
        noise_original = 0.08 * np.random.normal(0, 1, len(t))
        noise_reduced = 0.05 * np.random.normal(0, 1, len(t))
        noise_final = 0.03 * np.random.normal(0, 1, len(t))
        
        # 이상치 추가
        outlier_mask = np.random.random(len(t)) < 0.02
        outliers = np.where(outlier_mask, np.random.uniform(0.2, 0.5, len(t)), 0)
        
        demo_data['rms_original'] = base_trend + noise_original + outliers
        demo_data['rms_outlier_removed'] = base_trend + noise_reduced
        demo_data['rms_final_processed'] = base_trend + noise_final
        demo_data['rms_final_cleaned'] = remove_spikes_from_rms(demo_data['rms_final_processed'])
        
        fig1 = visualize_complete_signal_processing(demo_data)
        plt.show()
        
        fig2 = visualize_processing_stages_comparison()
        plt.show()

if __name__ == "__main__":
    main()
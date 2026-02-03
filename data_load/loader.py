import os
import glob
import numpy as np
from nptdms import TdmsFile

from config.settings import INTERVAL_MIN, DURATION_S, FS, VALIDATION_DIR, TRAIN_DIR

def preprocess_signal(x, threshold=3.0):
    """신호 전처리 함수 - 이상치 제거 및 필터링"""
    from scipy import signal
    
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

def load_and_process_bearing(bearing_id, is_validation=False, full_load=True, channels=None, cutoff_percent=50):
    """베어링 데이터 로드 및 처리
    Args:
        bearing_id (int): 베어링 ID (1-8)
        is_validation (bool): 검증 데이터 여부
        full_load (bool): 전체 데이터 로드 여부
        channels (list): 사용할 채널 리스트
        cutoff_percent (int): 부분 로드시 사용할 비율
    """
    if channels is None:
        channels = ['CH1','CH2','CH3','CH4']
        
    if is_validation:
        folder = os.path.join(VALIDATION_DIR, f"Validation{bearing_id}")
    else:
        folder = os.path.join(TRAIN_DIR, f"Train{bearing_id}")
    
    files = sorted(glob.glob(os.path.join(folder, "*.tdms")))
    if not files:
        raise FileNotFoundError(f"No TDMS files in {folder}")
    
    if not is_validation and not full_load:
        cutoff = int(len(files) * cutoff_percent / 100)
        files = files[:cutoff]
    
    channel_rms = {ch: [] for ch in channels}
    
    # 간단한 진행률 표시로 변경
    processed_count = 0
    total_files = len(files)
    
    for fp in files:
        try:
            td = TdmsFile.read(fp)
            grp = td.groups()[0].name
            vib = {ch.name: ch.data for ch in td[grp].channels()}
            
            for ch in channels:
                if ch in vib:
                    filt = preprocess_signal(vib[ch])
                    channel_rms[ch].append(np.sqrt(np.mean(filt**2)))
                else:
                    channel_rms[ch].append(0.0)
        except Exception as e:
            for ch in channels:
                channel_rms[ch].append(0.0)
        
        processed_count += 1
        # 25% 단위로만 진행률 표시
        if processed_count % max(1, total_files // 4) == 0 or processed_count == total_files:
            percent = (processed_count / total_files) * 100
            print(f"\r    처리 중: {percent:.0f}% ({processed_count}/{total_files})", end="", flush=True)
    
    print()
    
    # 스파이크 제거
    for ch in channels:
        channel_rms[ch] = remove_spikes_from_rms(channel_rms[ch])
    
    return channel_rms, len(files)

def load_all_bearing_data(test_id, train_ids, channels, cutoff_percent):
    """모든 베어링 데이터 구조화 로드
    Args:
        test_id (int): 검증 데이터 ID
        train_ids (list): 훈련 데이터 ID 리스트
        channels (list): 사용할 채널 리스트
        cutoff_percent (int): 부분 로드 비율
    """
    bearing_data = {}
    
    # 검증 데이터 로드
    bearing_data['validation'] = {}
    bearing_data['validation']['full'], _ = load_and_process_bearing(
        test_id, is_validation=True, full_load=True, channels=channels
    )
    
    # 훈련 데이터 로드
    for bid in train_ids:
        bearing_data[bid] = {}
        
        # 전체 데이터
        bearing_data[bid]['full'], _ = load_and_process_bearing(
            bid, is_validation=False, full_load=True, 
            channels=channels, cutoff_percent=cutoff_percent
        )
        
        # 부분 데이터
        bearing_data[bid]['part'], _ = load_and_process_bearing(
            bid, is_validation=False, full_load=False, 
            channels=channels, cutoff_percent=cutoff_percent
        )
    
    return bearing_data

def get_file_count(bearing_id, is_validation=False):
    """베어링 데이터의 파일 개수 확인
    Args:
        bearing_id (int): 베어링 ID
        is_validation (bool): 검증 데이터 여부
    """
    if is_validation:
        folder = os.path.join(VALIDATION_DIR, f"Validation{bearing_id}")
    else:
        folder = os.path.join(TRAIN_DIR, f"Train{bearing_id}")
    
    files = glob.glob(os.path.join(folder, "*.tdms"))
    return len(files)

def check_data_availability():
    """데이터 가용성 확인"""
    print("데이터 가용성 확인:")
    
    # 훈련 데이터 확인
    print("\n 훈련 데이터:")
    for i in range(1, 9):
        try:
            count = get_file_count(i, is_validation=False)
            print(f"   Train{i}: {count}개 파일")
        except:
            print(f"   Train{i}: ❌ 없음")
    
    # 검증 데이터 확인
    print("\n 검증 데이터:")
    for i in range(1, 7):
        try:
            count = get_file_count(i, is_validation=True)
            print(f"   Validation{i}: {count}개 파일")
        except:
            print(f"   Validation{i}: ❌ 없음")

def analyze_bearing_similarity(test_id, candidate_ids, bearing_data, channels):
    """베어링 유사도 분석 - 유사도 기반 가중치 설정"""
    similarities = {}
    
    val_rms = bearing_data['validation']['full']
    val_duration_hours = len(val_rms[channels[0]]) * INTERVAL_MIN / 60
    val_mat = np.column_stack([val_rms[ch] for ch in channels]).astype(np.float32)
    
    for bid in candidate_ids:
        train_rms = bearing_data[bid]['full']
        train_duration_hours = len(train_rms[channels[0]]) * INTERVAL_MIN / 60
        
        if train_duration_hours >= val_duration_hours:
            train_end_idx = int(len(train_rms[channels[0]]) * (val_duration_hours / train_duration_hours))
            train_mat = np.column_stack([train_rms[ch][:train_end_idx] for ch in channels]).astype(np.float32)
        else:
            train_mat = np.column_stack([train_rms[ch] for ch in channels]).astype(np.float32)
        
        min_len = min(len(val_mat), len(train_mat))
        val_slice = val_mat[:min_len]
        train_slice = train_mat[:min_len]
        
        correlation = np.mean([
            np.corrcoef(val_slice[:, i], train_slice[:, i])[0, 1]
            for i in range(len(channels))
        ])
        
        similarities[bid] = max(0, correlation)
    
    total = sum(similarities.values())
    if total > 0:
        weights = {bid: (sim/total * len(similarities)) for bid, sim in similarities.items()}
    else:
        weights = {bid: 1.0 for bid in similarities}
    
    return weights

if __name__ == "__main__":
    # 데이터 가용성 테스트
    check_data_availability()
    
    # 샘플 로드 테스트
    try:
        print("\n 샘플 로드 테스트:")
        rms_data, file_count = load_and_process_bearing(1, is_validation=False, full_load=False, cutoff_percent=10)
        print(f"✅ Train1 샘플 로드 성공: {file_count}개 파일")
        print(f"   채널별 RMS 개수: {[(ch, len(vals)) for ch, vals in rms_data.items()]}")
    except Exception as e:
        print(f"❌ 샘플 로드 실패: {e}")

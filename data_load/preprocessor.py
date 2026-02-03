"""
데이터 전처리 및 변환 기능
"""
import numpy as np
from scipy import signal

def normalize_data(data, method='minmax'):
    """데이터 정규화
    Args:
        data (array): 입력 데이터
        method (str): 정규화 방법 ('minmax', 'zscore')
    """
    data = np.array(data)
    
    if method == 'minmax':
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val > min_val:
            return (data - min_val) / (max_val - min_val)
        else:
            return np.zeros_like(data)
    
    elif method == 'zscore':
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val > 0:
            return (data - mean_val) / std_val
        else:
            return np.zeros_like(data)
    
    return data

def extract_statistical_features(signal_data):
    """통계적 특성 추출 """
    signal_data = np.array(signal_data)
    
    features = {
        'mean': np.mean(signal_data),
        'std': np.std(signal_data),
        'rms': np.sqrt(np.mean(signal_data**2)),
        'peak': np.max(np.abs(signal_data)),
        'crest_factor': np.max(np.abs(signal_data)) / np.sqrt(np.mean(signal_data**2)),
        'skewness': calculate_skewness(signal_data),
        'kurtosis': calculate_kurtosis(signal_data)
    }
    
    return features

def calculate_skewness(data):
    """왜도 계산"""
    data = np.array(data)
    mean_val = np.mean(data)
    std_val = np.std(data)
    
    if std_val == 0:
        return 0
    
    return np.mean(((data - mean_val) / std_val) ** 3)

def calculate_kurtosis(data):
    """첨도 계산"""
    data = np.array(data)
    mean_val = np.mean(data)
    std_val = np.std(data)
    
    if std_val == 0:
        return 0
    
    return np.mean(((data - mean_val) / std_val) ** 4) - 3

def apply_frequency_filter(data, fs, filter_type='highpass', cutoff=25, order=4):
    """주파수 필터 적용
    Args:
        data (array): 입력 데이터
        fs (float): 샘플링 주파수
        filter_type (str): 필터 타입
        cutoff (float): 차단 주파수
        order (int): 필터 차수
    """
    nyquist = fs / 2
    normalized_cutoff = cutoff / nyquist
    
    b, a = signal.butter(order, normalized_cutoff, btype=filter_type)
    filtered_data = signal.filtfilt(b, a, data)
    
    return filtered_data

def remove_power_line_noise(data, fs, frequencies=[50, 60], Q=30):
    """전력선 노이즈 제거 
    Args:
        data (array): 입력 데이터
        fs (float): 샘플링 주파수
        frequencies (list): 제거할 주파수들
        Q (float): 노치 필터 Q 팩터
    """
    filtered_data = data.copy()
    
    for freq in frequencies:
        w0 = freq / (fs / 2)
        if w0 < 1.0:  # 나이퀴스트 주파수보다 작을 때만 적용
            b, a = signal.iirnotch(w0, Q)
            filtered_data = signal.filtfilt(b, a, filtered_data)
    
    return filtered_data

def detect_outliers(data, method='iqr', threshold=1.5):
    """이상치 검출  
    Args:
        data (array): 입력 데이터
        method (str): 검출 방법 ('iqr', 'zscore')
        threshold (float): 임계값
    """
    data = np.array(data)
    
    if method == 'iqr':
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = np.where((data < lower_bound) | (data > upper_bound))[0]
    
    elif method == 'zscore':
        z_scores = np.abs((data - np.mean(data)) / np.std(data))
        outliers = np.where(z_scores > threshold)[0]
    
    return outliers

def smooth_signal(data, window_size=5, method='moving_average'):
    """신호 스무딩
    Args:
        data (array): 입력 데이터
        window_size (int): 윈도우 크기
        method (str): 스무딩 방법
    """
    data = np.array(data)
    
    if method == 'moving_average':
        if window_size > len(data):
            window_size = len(data)
        
        smoothed = np.convolve(data, np.ones(window_size)/window_size, mode='same')
        
    elif method == 'median':
        from scipy.ndimage import median_filter
        smoothed = median_filter(data, size=window_size)
    
    return smoothed

class DataPreprocessor:
    """데이터 전처리 클래스"""
    
    def __init__(self, fs=25600):
        self.fs = fs
        self.scaler_params = {}
    
    def fit_scaler(self, data, method='minmax'):
        """스케일러 파라미터 학습"""
        data = np.array(data)
        
        if method == 'minmax':
            self.scaler_params = {
                'method': 'minmax',
                'min': np.min(data),
                'max': np.max(data)
            }
        elif method == 'zscore':
            self.scaler_params = {
                'method': 'zscore',
                'mean': np.mean(data),
                'std': np.std(data)
            }
    
    def transform(self, data):
        """데이터 변환"""
        if not self.scaler_params:
            return data
        
        data = np.array(data)
        method = self.scaler_params['method']
        
        if method == 'minmax':
            min_val = self.scaler_params['min']
            max_val = self.scaler_params['max']
            if max_val > min_val:
                return (data - min_val) / (max_val - min_val)
        
        elif method == 'zscore':
            mean_val = self.scaler_params['mean']
            std_val = self.scaler_params['std']
            if std_val > 0:
                return (data - mean_val) / std_val
        
        return data
    
    def preprocess_pipeline(self, data, steps=['filter', 'denoise', 'normalize']):
        """전처리 파이프라인 실행"""
        processed_data = data.copy()
        
        for step in steps:
            if step == 'filter':
                processed_data = apply_frequency_filter(processed_data, self.fs)
            elif step == 'denoise':
                processed_data = remove_power_line_noise(processed_data, self.fs)
            elif step == 'normalize':
                processed_data = normalize_data(processed_data)
            elif step == 'smooth':
                processed_data = smooth_signal(processed_data)
        
        return processed_data

if __name__ == "__main__":
    # 테스트 코드
    test_data = np.random.randn(1000) + np.sin(np.linspace(0, 10*np.pi, 1000))
    
    print("🧪 전처리 기능 테스트:")
    
    # 통계적 특성 추출 테스트
    features = extract_statistical_features(test_data)
    print(f" 통계적 특성: {features}")
    
    # 정규화 테스트
    normalized = normalize_data(test_data)
    print(f" 정규화 완료: min={np.min(normalized):.3f}, max={np.max(normalized):.3f}")
    
    # 전처리 파이프라인 테스트
    preprocessor = DataPreprocessor()
    processed = preprocessor.preprocess_pipeline(test_data[:100])  # 샘플 데이터로 테스트
    print(f" 전처리 파이프라인 완료: shape={processed.shape}")

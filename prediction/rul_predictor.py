"""
RUL 예측 메인 로직
"""

import os
import re
import numpy as np
import torch

from config.settings import MODEL_DIR, INTERVAL_MIN
from config.group_config import GroupConfigManager
from data_load.loader import (
    load_and_process_bearing, 
    load_all_bearing_data,
    analyze_bearing_similarity
)
from features.health_index import (
    train_pca_model_from_data, 
    calculate_hi_with_trained_pca, 
    calculate_rul_from_hi,
    create_health_index
)
from models.transformers import TimeSeriesTransformer, TransformerRULModel
from prediction.ensemble import ensemble_forecast

class GroupBasedRULPredictor:
    """그룹 기반 RUL 예측 클래스"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config_manager = GroupConfigManager()
    
    def extract_validation_id(self, validation_name: str) -> int:
        """검증 폴더명에서 ID 추출"""
        match = re.search(r'(\d+)', validation_name)
        if not match:
            raise ValueError(f"검증 폴더명에서 ID를 추출할 수 없습니다: {validation_name}")
        return int(match.group(1))
    
    def predict_timeseriesTransformer(self, validation_id, config):
        """TimeSeriesTransformer 모델을 사용한 예측 - 유사도 기반 가중치 추가"""
        channels = config['channels']
        train_ids = config['train_ids']
        cutoff_percent = config['cutoff_percent']
        safety_factor = config['safety_factor']
        model_pattern = config['model_file_pattern']
        
        # 기존 코드와 동일한 데이터 구조화
        bearing_data = load_all_bearing_data(validation_id, train_ids, channels, cutoff_percent)
        
        # 유사도 기반 가중치 계산 (추가됨)
        train_weights = analyze_bearing_similarity(validation_id, train_ids, bearing_data, channels)
        
        train_data = []
        train_weights_list = []
        for bid in train_ids:
            mat = np.column_stack([bearing_data[bid]['full'][ch] for ch in channels])
            train_data.append(mat)
            train_weights_list.append(train_weights[bid])
        
        # 기존 코드와 동일한 복잡한 PCA 학습
        pca, scaler, avg_life, ref_hi_data = train_pca_model_from_data(
            train_data, train_ids, weights=train_weights_list)
        
        # Validation HI 계산
        val_hi = calculate_hi_with_trained_pca(
            bearing_data['validation']['full'], pca, scaler, channels
        )
        
        # 모델 로드
        models = []
        src_len = 3
        for i in range(1, 4):
            model = TimeSeriesTransformer(src_len)
            model_path = os.path.join(MODEL_DIR, model_pattern.format(i))
            
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                model.eval()
                models.append(model)
        
        # 현재 상태 계산
        current_time = len(val_hi) * INTERVAL_MIN / 60
        current_hi = val_hi[-1]
        
        # 미래 예측
        remaining_life_estimate = avg_life - current_time
        if remaining_life_estimate > 0:
            pred_hours = min(remaining_life_estimate * 1.5, avg_life)
        else:
            pred_hours = avg_life * 0.5
        
        pred_length = int(pred_hours * 60 / INTERVAL_MIN)
        seed_length = min(30, len(val_hi))
        val_hi_seed = val_hi[-seed_length:]
        
        if seed_length >= src_len and pred_length > 0 and models:
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
        
        # RUL 계산
        rul_pred = calculate_rul_from_hi(hi_pred, avg_life, safety_factor)
        predicted_rul = rul_pred[0] if len(rul_pred) > 0 else 0
        
        return {
            'validation_id': validation_id,
            'current_time': current_time,
            'predicted_rul': predicted_rul,
            'safety_factor': safety_factor,
            'model_type': 'TimeSeriesTransformer',
            'train_ids': train_ids
        }
    
    def predict_transformerRUL(self, validation_id, config):
        """TransformerRULModel을 사용한 예측"""
        channels = config['channels']
        train_ids = config['train_ids']
        safety_factor = config['safety_factor']
        model_pattern = config['model_file_pattern']
        
        # 학습 데이터로 스케일러와 PCA 생성
        train_rms, _ = load_and_process_bearing(
            train_ids[0], is_validation=False, full_load=False, channels=channels
        )
        train_hi, scaler, pca = create_health_index(train_rms, channels)
        
        train_length = len(train_hi)
        sequence_length = min(50, train_length // 2)
        if train_length < sequence_length:
            sequence_length = max(10, train_length - 1)
        
        # Validation 데이터 처리
        val_rms, _ = load_and_process_bearing(
            validation_id, is_validation=True, full_load=True, channels=channels
        )
        val_data_matrix = np.array([val_rms[ch] for ch in channels]).T
        val_data_normalized = scaler.transform(val_data_matrix)
        val_hi = pca.transform(val_data_normalized).flatten()
        
        # 모델 로드 및 예측
        predictions = []
        for i in range(1, 4):
            model_path = os.path.join(MODEL_DIR, model_pattern.format(i))
            if os.path.exists(model_path):
                model = TransformerRULModel(seq_len=sequence_length).to(self.device)
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.eval()
                
                with torch.no_grad():
                    if len(val_hi) >= sequence_length:
                        recent_hi = val_hi[-sequence_length:]
                    else:
                        recent_hi = np.pad(val_hi, (sequence_length - len(val_hi), 0), mode='edge')
                    
                    input_tensor = torch.FloatTensor(recent_hi).unsqueeze(0).unsqueeze(-1).to(self.device)
                    predicted_rul = model(input_tensor).item()
                    predicted_rul = max(predicted_rul, 0.1)
                    predictions.append(predicted_rul)
        
        # 결과 계산
        ensemble_rul = np.mean(predictions) if predictions else 10.0
        predicted_rul = ensemble_rul * safety_factor
        current_operating_time = len(val_hi) * INTERVAL_MIN / 60
        
        return {
            'validation_id': validation_id,
            'current_time': current_operating_time,
            'predicted_rul': predicted_rul,
            'safety_factor': safety_factor,
            'model_type': 'TransformerRULModel',
            'train_ids': train_ids
        }
    
    def predict_single_validation(self, validation_name, group_id, verbose=True):
        """단일 검증 데이터에 대한 RUL 예측"""
        if verbose:
            print(f"\n {validation_name} 처리 중 (그룹 {group_id})")
            print(f"    데이터 로딩 중...")  # 간단한 메시지로 변경
        
        try:
            # 검증 데이터 ID 추출
            validation_id = self.extract_validation_id(validation_name)
            
            # 그룹 및 검증 ID에 따른 설정 가져오기
            config = self.config_manager.get_config_for_validation(validation_id, group_id)
            
            if verbose:
                print(f"   설정: {config['model_type']}, Train IDs: {config['train_ids']}")
                print(f"   채널: {config['channels']}, Safety Factor: {config['safety_factor']}")
            
            # 모델 타입에 따른 예측 수행
            if config['model_type'] == 'TimeSeriesTransformer':
                result = self.predict_timeseriesTransformer(validation_id, config)
            elif config['model_type'] == 'TransformerRULModel':
                result = self.predict_transformerRUL(validation_id, config)
            else:
                raise ValueError(f"지원하지 않는 모델 타입: {config['model_type']}")
            
            result['group_id'] = group_id
            result['validation_name'] = validation_name
            result['success'] = True
            
            if verbose:
                print(f"   ✅ 예측 완료: {result['predicted_rul']:.2f}시간")
                print(f"   현재 운영시간: {result['current_time']:.2f}시간")
            
            return result
            
        except Exception as e:
            if verbose:
                print(f"   ❌ 예측 실패: {str(e)}")
            return {
                'validation_name': validation_name,
                'validation_id': validation_id,
                'group_id': group_id,
                'current_time': 0,
                'predicted_rul': 0,
                'error': str(e),
                'success': False
            }

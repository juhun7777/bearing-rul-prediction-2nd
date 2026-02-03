"""
그룹별 설정 관리
"""
class GroupConfigManager:
    def __init__(self):
        self.group_configs = {
            1: { # (TRAIN 1,3 사용)
                'channels': ['CH1', 'CH2', 'CH3', 'CH4'],
                'train_ids': [1,3],  
                'cutoff_percent': 50,
                'model_type': 'TimeSeriesTransformer',
                'model_file_pattern': 'Valid24_transformer_ensemble_{}.pt'  
            },
            3: {  #(TRAIN 4,5,7 사용)
                'channels': ['CH1', 'CH2', 'CH3', 'CH4'],
                'train_ids': [4,5,7], 
                'cutoff_percent': 50,
                'model_type': 'TimeSeriesTransformer',
                'model_file_pattern': 'Valid13_transformer_ensemble_{}.pt'  # 수정됨
            },
            4: {  # Validation 5,6 그룹 (TRAIN 6 사용)
                'channels': ['CH1', 'CH2', 'CH3'],
                'train_ids': [6],
                'cutoff_percent': 100,
                'model_type': 'TransformerRULModel',
                'model_file_pattern': 'Valid56_model_ensemble_{}.pt'
            }
        }
        
        #각 검증데이터별 안전계수 설정(검증 구간에서 학습폴더와의 에너지 방출 크기 비율)
        self.validation_safety_factors = {
            1: 1.1,
            2: 0.9,   
            3: 0.68,   
            4: 0.8,   
            5: 0.17,
            6: 0.45
        }
    
    def get_config_for_group(self, group_id):
        if group_id not in self.group_configs:
            # 기본 설정 반환 (safety_factor 제거)
            return {
                'channels': ['CH1', 'CH2', 'CH3', 'CH4'],
                'train_ids': [1, 3],
                'cutoff_percent': 50,
                'model_type': 'TimeSeriesTransformer',
                'model_file_pattern': 'Valid24_transformer_ensemble_{}.pt'
            }
        return self.group_configs[group_id]
    
    def get_config_for_validation(self, validation_id, group_id):
        """검증 데이터와 그룹에 따른 설정 반환"""
        # Validation 5, 6은 특별 처리
        if validation_id in [5, 6]:
            config = {
                'channels': ['CH1', 'CH2', 'CH3'],
                'train_ids': [6],
                'cutoff_percent': 100,
                'model_type': 'TransformerRULModel',
                'model_file_pattern': 'Valid56_model_ensemble_{}.pt'
            }
        else:
            # 그룹 설정 가져오기
            config = self.get_config_for_group(group_id).copy()
        
        # 검증 ID별 안전계수 추가
        config['safety_factor'] = self.validation_safety_factors.get(validation_id, 0.85)
        
        return config

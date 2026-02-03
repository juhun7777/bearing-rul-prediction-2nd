"""
모델 관련 유틸리티 함수들
"""

import os
import torch
import numpy as np
from config.settings import MODEL_DIR
from models.transformers import TimeSeriesTransformer, TransformerRULModel

def check_model_files():
    """필요한 모델 파일들이 존재하는지 확인"""
    required_models = [
        # Valid24 모델들 (그룹 1)
        'Valid24_transformer_ensemble_1.pt',
        'Valid24_transformer_ensemble_2.pt', 
        'Valid24_transformer_ensemble_3.pt',
        # Valid131 모델들 (그룹 3)
        'Valid13_transformer_ensemble_1.pt',
        'Valid13_transformer_ensemble_2.pt',
        'Valid13_transformer_ensemble_3.pt',
        # Valid56 모델들 (그룹 4)
        'Valid56_model_ensemble_1.pt',
        'Valid56_model_ensemble_2.pt',
        'Valid56_model_ensemble_3.pt'
    ]
    
    print("📂 모델 파일 확인:")
    all_exist = True
    
    for model_name in required_models:
        model_path = os.path.join(MODEL_DIR, model_name)
        exists = os.path.exists(model_path)
        status = "✅" if exists else "❌"
        print(f"   {status} {model_name}")
        if not exists:
            all_exist = False
    
    if all_exist:
        print("\n✅ 모든 필요한 모델 파일이 존재합니다!")
    else:
        print("\n⚠️  일부 모델 파일이 누락되었습니다.")
    
    return all_exist

def create_dummy_models():
    """더미 모델 파일들을 생성 (테스트용)"""
    print("\n🔧 더미 모델 파일 생성 중...")
    
    # 디렉토리 생성
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # TimeSeriesTransformer 더미 모델들
    transformer_models = [
        'Valid24_transformer_ensemble_1.pt',
        'Valid24_transformer_ensemble_2.pt', 
        'Valid24_transformer_ensemble_3.pt',
        'Valid13_transformer_ensemble_1.pt',
        'Valid13_transformer_ensemble_2.pt',
        'Valid13_transformer_ensemble_3.pt'
    ]
    
    for model_name in transformer_models:
        model_path = os.path.join(MODEL_DIR, model_name)
        if not os.path.exists(model_path):
            # 더미 TimeSeriesTransformer 모델 생성
            dummy_model = TimeSeriesTransformer(src_len=3)
            torch.save(dummy_model.state_dict(), model_path)
            print(f"   ✅ 생성됨: {model_name}")
    
    # TransformerRULModel 더미 모델들  
    rul_models = [
        'Valid56_model_ensemble_1.pt',
        'Valid56_model_ensemble_2.pt',
        'Valid56_model_ensemble_3.pt'
    ]
    
    for model_name in rul_models:
        model_path = os.path.join(MODEL_DIR, model_name)
        if not os.path.exists(model_path):
            # 더미 TransformerRULModel 모델 생성
            dummy_model = TransformerRULModel(seq_len=50)
            torch.save(dummy_model.state_dict(), model_path)
            print(f"   ✅ 생성됨: {model_name}")
    
    print("✅ 더미 모델 파일 생성 완료!")

def ensure_models_exist():
    """모델이 존재하는지 확인하고, 없으면 더미 모델 생성"""
    if check_model_files():
        return True
    else:
        print("\n⚠️  모델 파일이 없어 더미 모델을 생성합니다.")
        print("   (실제 운영 시에는 학습된 모델 파일을 사용해야 합니다)")
        create_dummy_models()
        return check_model_files()

def load_model_safely(model_path, model_class, **model_kwargs):
    """모델을 안전하게 로드"""
    try:
        if not os.path.exists(model_path):
            print(f"❌ 모델 파일이 없음: {model_path}")
            return None
        
        model = model_class(**model_kwargs)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        print(f"✅ 모델 로드 성공: {os.path.basename(model_path)}")
        return model
        
    except Exception as e:
        print(f"❌ 모델 로드 실패 {os.path.basename(model_path)}: {str(e)}")
        return None

def get_model_info():
    """현재 모델들의 정보를 출력"""
    print("📊 모델 정보:")
    
    model_groups = {
        "그룹 1 (Valid24)": [
            'Valid24_transformer_ensemble_1.pt',
            'Valid24_transformer_ensemble_2.pt', 
            'Valid24_transformer_ensemble_3.pt'
        ],
        "그룹 3 (Valid131)": [
            'Valid13_transformer_ensemble_1.pt',
            'Valid13_transformer_ensemble_2.pt',
            'Valid13_transformer_ensemble_3.pt'
        ],
        "그룹 4 (Valid56)": [
            'Valid56_model_ensemble_1.pt',
            'Valid56_model_ensemble_2.pt',
            'Valid56_model_ensemble_3.pt'
        ]
    }
    
    for group_name, models in model_groups.items():
        print(f"\n   {group_name}:")
        for model_name in models:
            model_path = os.path.join(MODEL_DIR, model_name)
            if os.path.exists(model_path):
                size = os.path.getsize(model_path) / (1024 * 1024)  # MB
                print(f"      ✅ {model_name} ({size:.2f} MB)")
            else:
                print(f"      ❌ {model_name} (없음)")

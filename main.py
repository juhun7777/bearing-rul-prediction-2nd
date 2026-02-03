"""
그룹 분류 기반 RUL 예측 시스템 - 메인 실행 파일 (모델 파일 확인 포함)
- 그룹 분류 클래스 결과를 기반으로 각 검증 데이터를 그룹별 설정으로 처리
"""

import warnings
warnings.filterwarnings('ignore')

# 모듈 import
from prediction.integrated_system import IntegratedGroupBasedSystem
from prediction.rul_predictor import GroupBasedRULPredictor
from grouping.classifier import BearingGroupClassifier
from utils.model_utils import ensure_models_exist

def main():
    """메인 실행 함수"""
    print(" 그룹 분류 기반 베어링 RUL 예측 시스템")
    print("="*80)
    
    # 1단계: 모델 파일 확인 및 준비
    print("\n 1단계: 모델 파일 확인")
    models_ready = ensure_models_exist()
    
    if not models_ready:
        print("❌ 모델 파일 준비 실패. 시스템을 종료합니다.")
        return None
    
    # 2단계: 통합 시스템 실행
    print("\n 2단계: 시스템 실행")
    print("   데이터 로딩 및 예측을 수행합니다...")
    system = IntegratedGroupBasedSystem()
    
    # 전체 시스템 실행
    simple_results, prediction_results = system.run_complete_system(verbose=True)
    
    # 제출 파일 생성
    submission_df = system.create_submission_file(prediction_results)
    
    print(f"\n 시스템 실행 완료!")
    
    return system, simple_results, prediction_results, submission_df

def test_model_files_only():
    """모델 파일만 확인하는 테스트"""
    print(" 모델 파일 확인 테스트")
    print("-" * 40)
    
    from utils.model_utils import check_model_files, create_dummy_models
    
    # 현재 상태 확인
    has_models = check_model_files()
    
    if not has_models:
        print("\n⚠️  모델 파일이 없어 더미 모델을 생성합니다.")
        create_dummy_models()
        check_model_files()
    
    return has_models

def test_group_classification_only():
    """그룹 분류만 테스트"""
    print(" 그룹 분류 테스트")
    print("-" * 40)
    
    classifier = BearingGroupClassifier()
    classification_results = classifier.classify_all_validations(verbose=True)
    simple_results = classifier.get_simple_results(classification_results)
    
    print(f"\n 간단한 결과:")
    for validation_name, result in simple_results.items():
        group = result.get('predicted_group', 'None')
        print(f"   {validation_name}: 그룹 {group}")
    
    return simple_results

def test_single_validation_prediction(validation_name="Validation1", group_id=1):
    """단일 검증 데이터 예측 테스트"""
    print(f" 단일 예측 테스트: {validation_name} (그룹 {group_id})")
    print("-" * 40)
    
    # 먼저 모델 파일 확인
    if not ensure_models_exist():
        print("❌ 모델 파일이 준비되지 않아 예측을 수행할 수 없습니다.")
        return None
    
    predictor = GroupBasedRULPredictor()
    result = predictor.predict_single_validation(validation_name, group_id, verbose=True)
    
    print(f"\n 예측 결과:")
    if result.get('success', False):
        print(f"   RUL: {result['predicted_rul']:.2f}시간 ({result['predicted_rul']*3600:.0f}초)")
        print(f"   현재 운영시간: {result['current_time']:.2f}시간")
        print(f"   모델 타입: {result.get('model_type', 'Unknown')}")
        print(f"   Train IDs: {result.get('train_ids', [])}")
        print(f"   안전계수: {result.get('safety_factor', 0.85)}")
    else:
        print(f"   오류: {result.get('error', '알 수 없는 오류')}")
    
    return result

if __name__ == "__main__":
    # 전체 시스템 실행
    result = main()
    
    if result is not None:
        system, simple_results, prediction_results, submission_df = result
    
    # 개별 테스트도 가능
    # test_model_files_only()
    # test_group_classification_only()
    # test_single_validation_prediction("Validation1", 1)

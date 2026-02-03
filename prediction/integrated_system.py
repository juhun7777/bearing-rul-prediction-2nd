"""
그룹 분류 + 그룹 기반 RUL 예측 통합 시스템
"""
import pandas as pd
from grouping.classifier import BearingGroupClassifier
from prediction.rul_predictor import GroupBasedRULPredictor

class IntegratedGroupBasedSystem:
    """그룹 분류 + 그룹 기반 RUL 예측 통합 시스템"""
    
    def __init__(self):
        self.classifier = BearingGroupClassifier()
        self.predictor = GroupBasedRULPredictor()
    
    def run_complete_system(self, verbose=True):
        """전체 시스템 실행"""
        if verbose:
            print("그룹 기반 베어링 RUL 예측 시스템 시작")
            print("="*80)
        
        # 1단계: 그룹 분류
        if verbose:
            print("\n1단계: 그룹 분류 수행")
            print("-" * 40)
        
        classification_results = self.classifier.classify_all_validations(verbose=verbose)
        simple_results = self.classifier.get_simple_results(classification_results)
        
        if verbose:
            print(f"\n✅ 그룹 분류 완료!")
            print(f"분류 결과:")
            for validation_name, result in simple_results.items():
                group = result.get('predicted_group', 'None')
                print(f"   {validation_name}: 그룹 {group}")
        
        # 2단계: 그룹 기반 RUL 예측
        if verbose:
            print(f"\n 2단계: 그룹 기반 RUL 예측 수행")
            print("-" * 40)
        
        prediction_results = {}
        
        for validation_name, classification_result in simple_results.items():
            predicted_group = classification_result.get('predicted_group')
            
            if predicted_group is None:
                if verbose:
                    print(f"\n❌ {validation_name}: 그룹이 분류되지 않음")
                prediction_results[validation_name] = {
                    'validation_name': validation_name,
                    'predicted_group': None,
                    'rul_seconds': 0,
                    'error': '그룹이 분류되지 않음',
                    'success': False
                }
                continue
            
            # 그룹별 RUL 예측
            result = self.predictor.predict_single_validation(
                validation_name, predicted_group, verbose=verbose
            )
            
            # 결과 저장 (시간을 초로 변환)
            if result.get('success', False):
                rul_seconds = result['predicted_rul'] * 3600
                prediction_results[validation_name] = {
                    'validation_name': validation_name,
                    'predicted_group': predicted_group,
                    'rul_seconds': rul_seconds,
                    'rul_hours': result['predicted_rul'],
                    'current_time_hours': result['current_time'],
                    'safety_factor': result.get('safety_factor', 0.85),
                    'model_type': result.get('model_type', 'Unknown'),
                    'train_ids': result.get('train_ids', []),
                    'success': True
                }
            else:
                prediction_results[validation_name] = {
                    'validation_name': validation_name,
                    'predicted_group': predicted_group,
                    'rul_seconds': 0,
                    'error': result.get('error', '알 수 없는 오류'),
                    'success': False
                }
        
        # 3단계: 최종 결과 요약
        if verbose:
            self.print_final_summary(simple_results, prediction_results)
        
        return simple_results, prediction_results
    
    def print_final_summary(self, simple_results, prediction_results):
        """최종 결과 요약 출력"""
        print(f"\n{'='*80}")
        print(f"최종 결과 요약")
        print(f"{'='*80}")
        
        # 그룹별 분류 현황
        print(f"\n 그룹별 분류 현황:")
        group_counts = {1: [], 2: [], 3: [], 4: [], None: []}
        for validation_name, result in simple_results.items():
            group = result.get('predicted_group')
            group_counts[group].append(validation_name)
        
        for group_id in [1, 2, 3, 4]:
            validations = group_counts[group_id]
            if validations:
                print(f"   그룹 {group_id}: {len(validations)}개 - {', '.join(validations)}")
        
        if group_counts[None]:
            print(f"   미분류: {len(group_counts[None])}개 - {', '.join(group_counts[None])}")
        
        # RUL 예측 결과
        print(f"\n RUL 예측 결과:")
        successful_predictions = []
        failed_predictions = []
        
        for validation_name, result in prediction_results.items():
            if result.get('success', False):
                successful_predictions.append((validation_name, result))
            else:
                failed_predictions.append((validation_name, result))
        
        if successful_predictions:
            print(f"✅ 성공한 예측: {len(successful_predictions)}개")
            print(f"{'Validation':<12} {'그룹':<4} {'모델타입':<20} {'Train IDs':<12} {'RUL(초)':<10} {'RUL(시간)':<10} {'안전계수':<8}")
            print("-" * 85)
            
            for validation_name, result in successful_predictions:
                group = result['predicted_group']
                model_type = result.get('model_type', 'Unknown')
                train_ids = str(result.get('train_ids', []))
                rul_seconds = result['rul_seconds']
                rul_hours = result['rul_hours']
                safety_factor = result.get('safety_factor', 0.85)
                
                print(f"{validation_name:<12} {group:<4} {model_type:<20} {train_ids:<12} {rul_seconds:<10.0f} {rul_hours:<10.2f} {safety_factor:<8.2f}")
        
        if failed_predictions:
            print(f"\n❌ 실패한 예측: {len(failed_predictions)}개")
            for validation_name, result in failed_predictions:
                error = result.get('error', '알 수 없는 오류')
                group = result.get('predicted_group', 'N/A')
                print(f"   {validation_name} (그룹: {group}): {error}")
        
        print(f"\n 전체 통계:")
        print(f"   - 총 검증 데이터: {len(prediction_results)}개")
        print(f"   - 성공한 예측: {len(successful_predictions)}개")
        print(f"   - 실패한 예측: {len(failed_predictions)}개")
        print(f"   - 성공률: {len(successful_predictions)/len(prediction_results):.1%}")
    
    def create_submission_file(self, prediction_results, output_path="group_based_submission.xlsx"):
        """제출용 파일 생성"""
        submission_data = []
        
        for validation_name, result in prediction_results.items():
            if result.get('success', False):
                rul_seconds = int(result['rul_seconds'])
            else:
                rul_seconds = 0
            
            submission_data.append({
                'File': validation_name,
                'RUL_Score': rul_seconds
            })
        
        # DataFrame 생성 및 정렬
        df = pd.DataFrame(submission_data)
        if not df.empty:
            df['sort_key'] = df['File'].str.extract('(\d+)').astype(int)
            df = df.sort_values('sort_key').drop('sort_key', axis=1).reset_index(drop=True)
        
        # Excel 파일 저장
        try:
            df.to_excel(output_path, index=False, engine='openpyxl')
            print(f"\n✅ 제출 파일 저장 완료: {output_path}")
            print(f" 제출 파일 내용:")
            print(df.to_string(index=False))
        except Exception as e:
            print(f" 파일 저장: {e}")
            return None
        
        return df

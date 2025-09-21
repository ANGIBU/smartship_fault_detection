# main.py

import pandas as pd
import numpy as np
import warnings
import sys
warnings.filterwarnings('ignore')

from config import Config
from data_processor import DataProcessor
from model_training import ModelTraining
from prediction import Prediction
from utils import setup_logging, memory_usage_check

def main():
    """메인 실행 함수"""
    logger = setup_logging()
    logger.info("시스템 시작")
    Config.create_directories()
    initial_memory = memory_usage_check()
    
    try:
        # 1. 데이터 전처리
        print("\n" + "=" * 50)
        print("1단계: 데이터 전처리")
        print("=" * 50)
        
        processor = DataProcessor()
        X_train, X_test, y_train, train_ids, test_ids = processor.get_processed_data(
            use_feature_selection=True,
            scaling_method='robust'
        )
        
        print(f"훈련 데이터 형태: {X_train.shape}")
        print(f"테스트 데이터 형태: {X_test.shape}")
        print(f"타겟 분포: {pd.Series(y_train).value_counts().sort_index().head(10).values}")
        
        # 데이터 크기에 따른 설정 조정
        adjustments = Config.adjust_for_data_size(len(X_train), X_train.shape[1])
        if adjustments:
            print(f"데이터 크기 기반 조정: {adjustments}")
        
        # 2. 시간 기반 검증 전략 설정
        print("\n" + "=" * 50)
        print("2단계: 시간 기반 검증 전략 설정")
        print("=" * 50)
        
        from sklearn.model_selection import train_test_split
        
        # 시간 기반 분할을 기본으로 사용
        validation_config = Config.get_cross_validation_strategy('time_based')
        
        n_samples = len(X_train)
        train_ratio = validation_config['train_ratio']
        val_ratio = validation_config['val_ratio']
        holdout_ratio = validation_config['holdout_ratio']
        
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        # 시간 순서 기반 분할
        X_train_time = X_train.iloc[:train_end]
        X_val_time = X_train.iloc[train_end:val_end]
        X_holdout = X_train.iloc[val_end:]
        
        y_train_time = y_train.iloc[:train_end]
        y_val_time = y_train.iloc[train_end:val_end]
        y_holdout = y_train.iloc[val_end:]
        
        print(f"시간 기반 분할:")
        print(f"  훈련 세트: {X_train_time.shape}")
        print(f"  검증 세트: {X_val_time.shape}")
        print(f"  홀드아웃 세트: {X_holdout.shape}")
        
        # 추가 계층화 검증 세트 생성 (비교용)
        stratified_config = Config.get_cross_validation_strategy('stratified')
        X_train_strat, X_val_strat, y_train_strat, y_val_strat = train_test_split(
            X_train, y_train,
            test_size=stratified_config['test_size'],
            random_state=stratified_config['random_state'],
            stratify=y_train
        )
        
        print(f"\n계층화 분할 (비교용):")
        print(f"  훈련 세트: {X_train_strat.shape}")
        print(f"  검증 세트: {X_val_strat.shape}")
        
        # 검증 세트 분포 확인
        val_distribution = pd.Series(y_val_time).value_counts().sort_index()
        print(f"\n시간 기반 검증 세트 클래스 분포 (상위 10개):")
        print(val_distribution.head(10))
        
        # 3. 모델 훈련 (시간 기반 검증 우선 적용)
        print("\n" + "=" * 50)
        print("3단계: 모델 훈련")
        print("=" * 50)
        
        trainer = ModelTraining()
        
        # 시간 기반 검증으로 모델 훈련
        models, best_model = trainer.train_all_models(
            X_train_time, y_train_time,
            X_val_time, y_val_time,
            use_optimization=True
        )
        
        print(f"훈련된 모델 개수: {len(models)}")
        if best_model is not None:
            print(f"최고 성능 모델: {type(best_model).__name__}")
        
        # 4. 검증 수행 (시간 기반 우선)
        print("\n" + "=" * 50)
        print("4단계: 시간 기반 검증 수행")
        print("=" * 50)
        
        validation_scores = {}
        prediction_results = {}
        
        # 시간 기반 검증 (주요 검증)
        if best_model is not None:
            time_predictor = Prediction(best_model)
            time_predictions = time_predictor.predict(X_val_time)
            time_metrics = time_predictor.validate_predictions(y_val_time)
            
            if time_metrics:
                print(f"시간 기반 검증 Macro F1: {time_metrics['macro_f1']:.4f}")
                validation_scores['time_based'] = time_metrics['macro_f1']
                prediction_results['time_based'] = {
                    'predictions': time_predictions,
                    'metrics': time_metrics
                }
        
        # 홀드아웃 검증 (최종 성능 확인)
        if best_model is not None:
            holdout_predictor = Prediction(best_model)
            holdout_predictions = holdout_predictor.predict(X_holdout)
            holdout_metrics = holdout_predictor.validate_predictions(y_holdout)
            
            if holdout_metrics:
                print(f"홀드아웃 검증 Macro F1: {holdout_metrics['macro_f1']:.4f}")
                validation_scores['holdout'] = holdout_metrics['macro_f1']
                prediction_results['holdout'] = {
                    'predictions': holdout_predictions,
                    'metrics': holdout_metrics
                }
        
        # 계층화 검증 (비교용)
        if best_model is not None:
            strat_predictor = Prediction(best_model)
            strat_predictions = strat_predictor.predict(X_val_strat)
            strat_metrics = strat_predictor.validate_predictions(y_val_strat)
            
            if strat_metrics:
                print(f"계층화 검증 Macro F1: {strat_metrics['macro_f1']:.4f}")
                validation_scores['stratified'] = strat_metrics['macro_f1']
                prediction_results['stratified'] = {
                    'predictions': strat_predictions,
                    'metrics': strat_metrics
                }
        
        # 검증 결과 안정성 분석
        if len(validation_scores) >= 2:
            print(f"\n검증 점수 안정성 분석:")
            scores = list(validation_scores.values())
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            print(f"  평균 검증 점수: {mean_score:.4f}")
            print(f"  검증 점수 표준편차: {std_score:.4f}")
            print(f"  안정성 지수: {1 - std_score:.4f}")
            
            # 과적합 경고
            if std_score > 0.05:
                print(f"  경고: 검증 점수 불안정 (표준편차 > 0.05)")
                print(f"  권장사항: 시간 기반 검증 결과 우선 사용")
            
            # 시간 기반 검증과 다른 검증의 차이 분석
            if 'time_based' in validation_scores:
                time_score = validation_scores['time_based']
                for val_type, score in validation_scores.items():
                    if val_type != 'time_based':
                        diff = abs(time_score - score)
                        print(f"  시간 기반 vs {val_type}: 차이 {diff:.4f}")
                        if diff > 0.05:
                            print(f"    주의: {val_type} 검증에서 과적합 위험")
        
        # 5. 최종 모델 훈련 (안정적인 파라미터 사용)
        print("\n" + "=" * 50)
        print("5단계: 최종 모델 훈련")
        print("=" * 50)
        
        # 시간 기반 검증 결과가 가장 신뢰할 만하므로 이를 기준으로 최종 모델 훈련
        final_trainer = ModelTraining()
        
        # 안정적인 모델로 전체 데이터 재훈련
        print("안정적인 파라미터로 최종 모델 훈련")
        final_models, final_best_model = final_trainer.train_stable_models(X_train, y_train)
        
        if final_best_model is not None:
            final_predictor = Prediction(final_best_model)
            test_predictions = final_predictor.predict(X_test)
            distribution_info = final_predictor.analyze_prediction_distribution()
        elif best_model is not None:
            # 기존 모델 사용
            predictor = Prediction(best_model)
            test_predictions = predictor.predict(X_test)
            distribution_info = predictor.analyze_prediction_distribution()
            final_predictor = predictor
        else:
            print("경고: 사용 가능한 모델이 없습니다.")
            return None
        
        # 6. 제출 파일 생성
        print("\n" + "=" * 50)
        print("6단계: 제출 파일 생성")
        print("=" * 50)
        
        submission_df = final_predictor.create_submission_file(
            test_ids,
            apply_balancing=True
        )
        
        if submission_df is not None:
            print(f"제출 파일 생성 완료: {Config.RESULT_FILE}")
            print(f"제출 파일 형태: {submission_df.shape}")
        
        # 7. 성능 분석 및 최종 보고서
        print("\n" + "=" * 50)
        print("7단계: 성능 분석")
        print("=" * 50)
        
        # 교차 검증 결과 출력 (안정성 기준 정렬)
        if trainer.cv_scores:
            print("\n교차 검증 결과 (안정성 기준)")
            sorted_scores = sorted(trainer.cv_scores.items(),
                                 key=lambda x: x[1].get('stability_score', x[1]['mean']),
                                 reverse=True)
            for model_name, scores in sorted_scores:
                stability_score = scores.get('stability_score', scores['mean'])
                std_score = scores['std']
                print(f"{model_name:20s}: {stability_score:.4f} (표준편차: {std_score:.4f})")
        
        # 최종 성능 예측
        print(f"\n최종 성능 예측")
        if validation_scores:
            # 시간 기반 검증을 가장 신뢰할 만한 지표로 사용
            time_based_score = validation_scores.get('time_based', 0)
            if time_based_score > 0:
                # 보수적 성능 예측 (시간 기반 검증 - 안전 마진)
                conservative_estimate = time_based_score * 0.95
                print(f"시간 기반 검증 점수: {time_based_score:.4f}")
                print(f"보수적 성능 예측: {conservative_estimate:.4f}")
                print(f"목표 성능 (0.80)까지: {0.80 - conservative_estimate:.4f}점 필요")
                
                # 성능 방향 제시
                if conservative_estimate < 0.65:
                    print("권장사항: 피처 엔지니어링 및 데이터 품질 점검")
                elif conservative_estimate < 0.72:
                    print("권장사항: 모델 복잡도 조정 및 앙상블 기법 적용")
                elif conservative_estimate < 0.78:
                    print("권장사항: 하이퍼파라미터 세밀 조정")
                else:
                    print("권장사항: 현재 접근 방법 유지")
        
        # 클래스별 성능 분석 (안전한 접근 방식)
        if 'time_based' in prediction_results:
            time_metrics = prediction_results['time_based']['metrics']
            if time_metrics and 'class_metrics' in time_metrics:
                class_metrics = time_metrics['class_metrics']
                if class_metrics and len(class_metrics) > 0:
                    # 안전한 키 접근
                    low_performance_classes = []
                    for cm in class_metrics:
                        if isinstance(cm, dict):
                            # 다양한 키 패턴 시도
                            f1_score = cm.get('f1_score', cm.get('f1', cm.get('f1_score', 0)))
                            if f1_score < 0.4:
                                low_performance_classes.append(cm)
                    
                    if low_performance_classes:
                        print(f"\n성능이 낮은 클래스 ({len(low_performance_classes)}개):")
                        for i, cm in enumerate(low_performance_classes[:5]):
                            class_id = cm.get('class', i)
                            f1_score = cm.get('f1_score', cm.get('f1', 0))
                            support = cm.get('support', 0)
                            print(f"  클래스 {class_id:2d}: F1={f1_score:.3f}, 지원={support:4d}")
                        
                        if len(low_performance_classes) > 5:
                            print(f"  ... (총 {len(low_performance_classes)}개)")
        
        # 시스템 리소스 사용량 분석
        final_memory = memory_usage_check()
        memory_increase = final_memory - initial_memory
        
        print("\n" + "=" * 60)
        print("시스템 실행 완료")
        print("=" * 60)
        print(f"최종 예측 파일: {Config.RESULT_FILE}")
        print(f"최고 모델 파일: {Config.MODEL_FILE}")
        print(f"초기 메모리: {initial_memory:.2f} MB")
        print(f"최종 메모리: {final_memory:.2f} MB")
        print(f"메모리 증가량: {memory_increase:.2f} MB")
        
        # 검증 점수 요약
        if validation_scores:
            for val_type, score in validation_scores.items():
                print(f"{val_type} 검증 점수: {score:.4f}")
        
        if trainer.cv_scores:
            best_cv_scores = [score.get('stability_score', score['mean']) 
                             for score in trainer.cv_scores.values()]
            best_cv_score = max(best_cv_scores)
            print(f"최고 안정성 점수: {best_cv_score:.4f}")
        
        logger.info("시스템 정상 완료")
        
        # 반환값 구성
        result_dict = {
            'models': models,
            'best_model': best_model,
            'final_model': final_best_model,
            'validation_scores': validation_scores,
            'cv_scores': trainer.cv_scores,
            'distribution_info': distribution_info,
            'submission_df': submission_df,
            'prediction_results': prediction_results,
            'memory_usage': {
                'initial': initial_memory,
                'final': final_memory,
                'increase': memory_increase
            }
        }
        
        return result_dict
        
    except Exception as e:
        logger.error(f"시스템 실행 중 오류 발생: {e}")
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
        raise

def run_stable_mode():
    """안정적인 실행 모드"""
    print("=" * 50)
    print("   안정적인 실행 모드")
    print("=" * 50)
    
    try:
        Config.create_directories()
        
        # 데이터 전처리
        processor = DataProcessor()
        X_train, X_test, y_train, train_ids, test_ids = processor.get_processed_data(
            use_feature_selection=True,
            scaling_method='robust'
        )
        
        # 시간 기반 검증 데이터 분할
        n_samples = len(X_train)
        train_end = int(n_samples * 0.7)
        val_end = int(n_samples * 0.85)
        
        X_train_stable = X_train.iloc[:train_end]
        X_val_stable = X_train.iloc[train_end:val_end]
        X_holdout_stable = X_train.iloc[val_end:]
        
        y_train_stable = y_train.iloc[:train_end]
        y_val_stable = y_train.iloc[train_end:val_end]
        y_holdout_stable = y_train.iloc[val_end:]
        
        print(f"안정적인 분할:")
        print(f"  훈련 세트: {X_train_stable.shape}")
        print(f"  검증 세트: {X_val_stable.shape}")
        print(f"  홀드아웃 세트: {X_holdout_stable.shape}")
        
        # 안정적인 모델 훈련
        trainer = ModelTraining()
        models, best_model = trainer.train_stable_models(X_train_stable, y_train_stable)
        
        print(f"훈련된 모델 개수: {len(models)}")
        if best_model is not None:
            print(f"최고 성능 모델: {type(best_model).__name__}")
        
        # 검증 수행
        if best_model is not None:
            predictor = Prediction(best_model)
            
            # 홀드아웃 검증
            holdout_predictions = predictor.predict(X_holdout_stable)
            holdout_metrics = predictor.validate_predictions(y_holdout_stable)
            
            if holdout_metrics:
                print(f"홀드아웃 검증 Macro F1: {holdout_metrics['macro_f1']:.4f}")
            
            # 테스트 예측
            test_predictions = predictor.predict(X_test)
            submission_df = predictor.create_submission_file(test_ids, apply_balancing=True)
        else:
            submission_df = None
        
        print(f"안정적인 실행 완료: {Config.RESULT_FILE}")
        
        return models, best_model, submission_df
        
    except Exception as e:
        print(f"안정적인 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        raise

def run_fast_mode():
    """빠른 실행 모드"""
    print("=" * 50)
    print("   빠른 실행 모드")
    print("=" * 50)
    
    try:
        Config.create_directories()
        
        # 데이터 전처리
        processor = DataProcessor()
        X_train, X_test, y_train, train_ids, test_ids = processor.get_processed_data(
            use_feature_selection=True,
            scaling_method='robust'
        )
        
        # 단순 분할
        from sklearn.model_selection import train_test_split
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train,
            test_size=0.2,
            random_state=Config.RANDOM_STATE,
            stratify=y_train
        )
        
        print(f"훈련 세트: {X_train_split.shape}")
        print(f"검증 세트: {X_val.shape}")
        
        # 빠른 모델 훈련
        trainer = ModelTraining()
        models, best_model = trainer.train_all_models(
            X_train_split, y_train_split,
            X_val, y_val,
            use_optimization=False
        )
        
        print(f"훈련된 모델 개수: {len(models)}")
        if best_model is not None:
            print(f"최고 성능 모델: {type(best_model).__name__}")
        
        # 테스트 예측
        if best_model is not None:
            predictor = Prediction(best_model)
            test_predictions = predictor.predict(X_test)
            submission_df = predictor.create_submission_file(test_ids, apply_balancing=True)
        else:
            submission_df = None
        
        print(f"빠른 실행 완료: {Config.RESULT_FILE}")
        
        return models, best_model, submission_df
        
    except Exception as e:
        print(f"빠른 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "stable":
            run_stable_mode()
        elif mode == "fast":
            run_fast_mode()
        else:
            print("사용법:")
            print("  python main.py         # 전체 실행")
            print("  python main.py stable  # 안정적인 실행")
            print("  python main.py fast    # 빠른 실행")
    else:
        main()
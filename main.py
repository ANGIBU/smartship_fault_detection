# main.py

import pandas as pd
import numpy as np
import warnings
import sys
import time
import psutil
warnings.filterwarnings('ignore')

from config import Config
from data_processor import DataProcessor
from model_training import ModelTraining
from prediction import PredictionProcessor
from utils import setup_logging, memory_usage_check, timer
from sklearn.model_selection import train_test_split

def main():
    """메인 실행 함수"""
    start_time = time.time()
    logger = setup_logging()
    logger.info("시스템 시작")
    Config.create_directories()
    initial_memory = memory_usage_check()
    
    try:
        # 하드웨어 사양에 맞춘 설정 조정
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        cpu_cores = psutil.cpu_count()
        Config.update_for_hardware(available_memory_gb, cpu_cores)
        
        print(f"시스템 설정:")
        print(f"  가용 메모리: {available_memory_gb:.1f}GB")
        print(f"  CPU 코어: {cpu_cores}개")
        print(f"  작업 프로세스: {Config.N_JOBS}개")
        
        # 설정 검증
        config_errors = Config.validate_config()
        if config_errors:
            print("설정 오류 발견:")
            for error in config_errors:
                print(f"  - {error}")
            return None
        
        # 1. 데이터 전처리
        print("\n" + "=" * 50)
        print("1단계: 데이터 전처리")
        print("=" * 50)
        
        processor = DataProcessor()
        X_train, X_test, y_train, train_ids, test_ids = processor.get_processed_data(
            use_resampling=True,
            scaling_method=Config.SCALING_METHOD
        )
        
        print(f"훈련 데이터 형태: {X_train.shape}")
        print(f"테스트 데이터 형태: {X_test.shape}")
        print(f"타겟 분포 (상위 10개): {pd.Series(y_train).value_counts().sort_index().head(10).to_dict()}")
        
        # 2. 검증 전략 설정
        print("\n" + "=" * 50)
        print("2단계: 검증 전략 설정")
        print("=" * 50)
        
        # Stratified 분할로 검증 세트 생성
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train,
            test_size=Config.VALIDATION_SIZE,
            random_state=Config.RANDOM_STATE,
            stratify=y_train
        )
        
        print(f"Stratified 분할:")
        print(f"  훈련 세트: {X_train_split.shape}")
        print(f"  검증 세트: {X_val.shape}")
        
        # 검증 세트 분포 확인
        val_distribution = pd.Series(y_val).value_counts().sort_index()
        print(f"\n검증 세트 클래스 분포 (상위 10개):")
        for class_id, count in val_distribution.head(10).items():
            percentage = (count / len(y_val)) * 100
            print(f"  클래스 {class_id}: {count}개 ({percentage:.1f}%)")
        
        # 3. 모델 훈련
        print("\n" + "=" * 50)
        print("3단계: 모델 훈련")
        print("=" * 50)
        
        trainer = ModelTraining()
        
        # 확률 보정을 위한 작은 검증 세트 분리
        X_train_final, X_calibration, y_train_final, y_calibration = train_test_split(
            X_train_split, y_train_split,
            test_size=0.1,
            random_state=Config.RANDOM_STATE,
            stratify=y_train_split
        )
        
        print(f"최종 훈련 세트: {X_train_final.shape}")
        print(f"보정 세트: {X_calibration.shape}")
        
        # 모델 훈련
        models, best_model = trainer.train_all_models(
            X_train_final, y_train_final,
            X_calibration, y_calibration,
            use_optimization=True
        )
        
        print(f"훈련된 모델 개수: {len(models)}")
        if best_model is not None:
            print(f"최고 성능 모델: {type(best_model).__name__}")
        
        # 4. 검증 수행
        print("\n" + "=" * 50)
        print("4단계: 모델 검증")
        print("=" * 50)
        
        validation_results = {}
        
        if best_model is not None:
            predictor = PredictionProcessor(best_model)
            
            # 모델 확률 보정
            predictor.calibrate_model(X_calibration, y_calibration)
            
            # 검증 세트 예측
            val_predictions = predictor.predict(X_val, use_calibrated=True)
            val_metrics = predictor.validate_predictions(y_val)
            
            if val_metrics:
                print(f"검증 Macro F1: {val_metrics['macro_f1']:.4f}")
                validation_results['validation'] = val_metrics
                
                # 클래스별 성능 분석
                low_performance_classes = val_metrics.get('low_performance_classes', [])
                if low_performance_classes:
                    print(f"\n성능이 낮은 클래스: {len(low_performance_classes)}개")
                    for cls_info in low_performance_classes[:5]:
                        print(f"  클래스 {cls_info['class']}: F1={cls_info['f1_score']:.3f}")
        
        # 5. 테스트 예측
        print("\n" + "=" * 50)
        print("5단계: 테스트 예측")
        print("=" * 50)
        
        if best_model is not None:
            # 전체 훈련 데이터로 최종 모델 재훈련
            final_trainer = ModelTraining()
            
            # 최고 성능 모델과 동일한 타입으로 재훈련
            best_model_type = type(best_model).__name__
            print(f"전체 데이터로 {best_model_type} 재훈련")
            
            if 'LGBMClassifier' in best_model_type:
                final_model = final_trainer.train_lightgbm(X_train, y_train)
            elif 'XGBClassifier' in best_model_type:
                final_model = final_trainer.train_xgboost(X_train, y_train)
            elif 'CatBoost' in best_model_type:
                final_model = final_trainer.train_catboost(X_train, y_train)
            elif 'RandomForest' in best_model_type:
                final_model = final_trainer.train_random_forest(X_train, y_train)
            elif 'ExtraTrees' in best_model_type:
                final_model = final_trainer.train_extra_trees(X_train, y_train)
            else:
                # 앙상블인 경우 기존 모델 사용
                final_model = best_model
            
            if final_model is not None:
                final_predictor = PredictionProcessor(final_model)
                test_predictions = final_predictor.predict(X_test, use_calibrated=False)
                distribution_info = final_predictor.analyze_prediction_distribution()
            else:
                final_predictor = predictor
                test_predictions = predictor.predict(X_test, use_calibrated=True)
                distribution_info = predictor.analyze_prediction_distribution()
        else:
            print("사용 가능한 모델이 없습니다")
            return None
        
        # 6. 제출 파일 생성
        print("\n" + "=" * 50)
        print("6단계: 제출 파일 생성")
        print("=" * 50)
        
        submission_df = final_predictor.create_submission_file(
            test_ids,
            apply_balancing=True,
            confidence_threshold=0.6
        )
        
        if submission_df is not None:
            print(f"제출 파일 생성 완료: {Config.RESULT_FILE}")
            print(f"제출 파일 형태: {submission_df.shape}")
        
        # 7. 성능 분석
        print("\n" + "=" * 50)
        print("7단계: 성능 분석")
        print("=" * 50)
        
        # 교차 검증 결과 출력
        if trainer.cv_scores:
            print("\n교차 검증 결과 (안정성 순위):")
            sorted_scores = sorted(trainer.cv_scores.items(),
                                 key=lambda x: x[1].get('stability', x[1]['mean']),
                                 reverse=True)
            for model_name, scores in sorted_scores:
                stability_score = scores.get('stability', scores['mean'])
                std_score = scores['std']
                print(f"  {model_name:15s}: {stability_score:.4f} (±{std_score:.4f})")
        
        # 피처 중요도 분석
        feature_importance = trainer.get_feature_importance()
        if feature_importance:
            print("\n피처 중요도 (상위 모델):")
            for model_name, importance in list(feature_importance.items())[:2]:
                if len(importance) > 0:
                    print(f"  {model_name}: 평균 중요도 {np.mean(importance):.4f}")
        
        # 최종 성능 예측
        print(f"\n성능 예측:")
        if validation_results and 'validation' in validation_results:
            val_score = validation_results['validation']['macro_f1']
            conservative_estimate = val_score * 0.95
            print(f"  검증 Macro F1: {val_score:.4f}")
            print(f"  예상 성능: {conservative_estimate:.4f}")
            print(f"  목표 대비: {0.83 - conservative_estimate:.4f}점 {'달성' if conservative_estimate >= 0.83 else '부족'}")
        
        # 시스템 리소스 사용량
        final_memory = memory_usage_check()
        memory_increase = final_memory - initial_memory
        total_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("시스템 실행 완료")
        print("=" * 60)
        print(f"최종 예측 파일: {Config.RESULT_FILE}")
        print(f"최고 모델 파일: {Config.MODEL_FILE}")
        print(f"초기 메모리: {initial_memory:.2f} MB")
        print(f"최종 메모리: {final_memory:.2f} MB")
        print(f"메모리 증가량: {memory_increase:.2f} MB")
        print(f"총 실행 시간: {total_time/60:.1f}분")
        
        # 검증 점수 요약
        if validation_results:
            for val_type, result in validation_results.items():
                if 'macro_f1' in result:
                    print(f"{val_type} Macro F1: {result['macro_f1']:.4f}")
        
        if trainer.cv_scores:
            best_cv_scores = [score.get('stability', score['mean']) 
                             for score in trainer.cv_scores.values()]
            best_cv_score = max(best_cv_scores) if best_cv_scores else 0
            print(f"최고 안정성 점수: {best_cv_score:.4f}")
        
        logger.info("시스템 정상 완료")
        
        # 반환값 구성
        result_dict = {
            'models': models,
            'best_model': best_model,
            'final_model': final_model if 'final_model' in locals() else best_model,
            'validation_results': validation_results,
            'cv_scores': trainer.cv_scores,
            'distribution_info': distribution_info,
            'submission_df': submission_df,
            'feature_importance': feature_importance,
            'memory_usage': {
                'initial': initial_memory,
                'final': final_memory,
                'increase': memory_increase
            },
            'execution_time': total_time
        }
        
        return result_dict
        
    except Exception as e:
        logger.error(f"시스템 실행 중 오류 발생: {e}")
        print(f"오류 발생: {e}")
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
        
        # 기본 설정으로 데이터 전처리
        processor = DataProcessor()
        X_train, X_test, y_train, train_ids, test_ids = processor.get_processed_data(
            use_resampling=False,
            scaling_method='robust'
        )
        
        # 단순 분할
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train,
            test_size=0.2,
            random_state=Config.RANDOM_STATE,
            stratify=y_train
        )
        
        print(f"훈련 세트: {X_train_split.shape}")
        print(f"검증 세트: {X_val.shape}")
        
        # 기본 모델 훈련 (튜닝 비활성화)
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
            predictor = PredictionProcessor(best_model)
            test_predictions = predictor.predict(X_test, use_calibrated=False)
            submission_df = predictor.create_submission_file(
                test_ids, apply_balancing=True
            )
        else:
            submission_df = None
        
        print(f"빠른 실행 완료: {Config.RESULT_FILE}")
        
        return models, best_model, submission_df
        
    except Exception as e:
        print(f"빠른 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        raise

def run_validation_mode():
    """검증 모드"""
    print("=" * 50)
    print("   검증 모드")
    print("=" * 50)
    
    try:
        Config.create_directories()
        
        # 데이터 전처리
        processor = DataProcessor()
        X_train, X_test, y_train, train_ids, test_ids = processor.get_processed_data(
            use_resampling=True,
            scaling_method=Config.SCALING_METHOD
        )
        
        # 검증용 분할
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train,
            test_size=0.3,
            random_state=Config.RANDOM_STATE,
            stratify=y_train
        )
        
        print(f"검증용 훈련 세트: {X_train_split.shape}")
        print(f"검증 세트: {X_val.shape}")
        
        # 모델 훈련
        trainer = ModelTraining()
        models, best_model = trainer.train_all_models(
            X_train_split, y_train_split,
            use_optimization=True
        )
        
        # 검증 수행
        if best_model is not None:
            predictor = PredictionProcessor(best_model)
            val_predictions = predictor.predict(X_val, use_calibrated=False)
            val_metrics = predictor.validate_predictions(y_val)
            
            if val_metrics:
                print(f"검증 Macro F1: {val_metrics['macro_f1']:.4f}")
        
        print(f"검증 모드 완료")
        
        return val_metrics if 'val_metrics' in locals() else None
        
    except Exception as e:
        print(f"검증 모드 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "fast":
            run_fast_mode()
        elif mode == "validation":
            run_validation_mode()
        else:
            print("사용법:")
            print("  python main.py           # 전체 실행")
            print("  python main.py fast      # 빠른 실행")
            print("  python main.py validation # 검증 모드")
    else:
        main()
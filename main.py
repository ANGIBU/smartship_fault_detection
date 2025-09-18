# main.py

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from config import Config
from data_processor import DataProcessor
from model_training import ModelTraining
from prediction import Prediction
from utils import setup_logging, memory_usage_check

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("   스마트 장비 이상신호 감지 시스템")
    print("=" * 60)
    
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
            use_feature_selection=True
        )
        
        print(f"훈련 데이터 형태: {X_train.shape}")
        print(f"테스트 데이터 형태: {X_test.shape}")
        print(f"타겟 분포: {pd.Series(y_train).value_counts().sort_index().values}")
        
        # 2. 모델 훈련
        print("\n" + "=" * 50)
        print("2단계: 모델 훈련")
        print("=" * 50)
        
        trainer = ModelTraining()
        
        # 검증 데이터 분할
        from sklearn.model_selection import train_test_split
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, 
            test_size=Config.VALIDATION_SIZE, 
            random_state=Config.RANDOM_STATE,
            stratify=y_train
        )
        
        print(f"훈련 세트: {X_train_split.shape}")
        print(f"검증 세트: {X_val.shape}")
        
        # 모델 훈련 (효율적인 모델만)
        models, best_model = trainer.train_all_models(
            X_train_split, y_train_split, 
            X_val, y_val,
            use_optimization=True,
            model_list=['lightgbm', 'xgboost', 'random_forest']
        )
        
        print(f"훈련된 모델 개수: {len(models)}")
        print(f"최고 성능 모델: {type(best_model).__name__}")
        
        # 검증 세트 평가
        predictor_val = Prediction(best_model)
        val_predictions = predictor_val.predict(X_val)
        val_metrics = predictor_val.validate_predictions(y_val)
        
        if val_metrics:
            print(f"검증 세트 Macro F1 Score: {val_metrics['macro_f1']:.4f}")
        
        # 3. 테스트 예측
        print("\n" + "=" * 50)
        print("3단계: 테스트 예측")
        print("=" * 50)
        
        predictor = Prediction(best_model)
        
        # 단일 모델 예측
        test_predictions = predictor.predict(X_test)
        
        # 앙상블 예측 (성능 모델이 2개 이상인 경우)
        good_models = {name: model for name, model in models.items() 
                      if name in trainer.cv_scores and trainer.cv_scores[name]['mean'] >= 0.75}
        
        if len(good_models) >= 2:
            print(f"앙상블 사용 모델: {list(good_models.keys())}")
            ensemble_predictions = predictor.predict_with_ensemble(
                good_models, X_test, method='weighted_average'
            )
            
            # 예측 분포 균형 조정
            predictor.predictions = ensemble_predictions
        
        # 예측 분포 분석
        distribution_info = predictor.analyze_prediction_distribution()
        
        # 4. 제출 파일 생성
        print("\n" + "=" * 50)
        print("4단계: 제출 파일 생성")
        print("=" * 50)
        
        submission_df = predictor.create_submission_file(test_ids)
        
        print(f"제출 파일 생성 완료: {Config.RESULT_FILE}")
        print(f"제출 파일 형태: {submission_df.shape}")
        
        # 5. 모델 분석
        print("\n" + "=" * 50)
        print("5단계: 모델 분석")
        print("=" * 50)
        
        # 피처 중요도 분석
        feature_importance = trainer.feature_importance_analysis(
            best_model, 
            X_train.columns.tolist()
        )
        
        # CV 결과 출력
        if trainer.cv_scores:
            print("\n=== 교차 검증 결과 ===")
            for model_name, scores in trainer.cv_scores.items():
                print(f"{model_name}: {scores['mean']:.4f} (+/- {scores['std']*2:.4f})")
        
        # 최종 메모리 사용량
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
        
        if val_metrics:
            print(f"최종 검증 점수: {val_metrics['macro_f1']:.4f}")
        
        logger.info("시스템 정상 완료")
        
        return {
            'models': models,
            'best_model': best_model,
            'val_metrics': val_metrics,
            'distribution_info': distribution_info,
            'submission_df': submission_df
        }
        
    except Exception as e:
        logger.error(f"시스템 실행 중 오류 발생: {e}")
        print(f"오류 발생: {e}")
        raise

def run_prediction_only():
    """훈련된 모델로만 예측 수행"""
    print("=" * 50)
    print("   예측 전용 모드")
    print("=" * 50)
    
    try:
        Config.create_directories()
        
        processor = DataProcessor()
        X_train, X_test, y_train, train_ids, test_ids = processor.get_processed_data(
            use_feature_selection=True
        )
        
        predictor = Prediction()
        predictor.load_trained_model()
        
        predictions = predictor.predict(X_test)
        
        submission_df = predictor.create_submission_file(test_ids)
        
        print(f"예측 완료: {Config.RESULT_FILE}")
        
        return submission_df
        
    except Exception as e:
        print(f"예측 중 오류 발생: {e}")
        raise

def run_hyperparameter_optimization():
    """하이퍼파라미터 최적화 모드"""
    print("=" * 50)
    print("   하이퍼파라미터 최적화 모드")
    print("=" * 50)
    
    try:
        Config.create_directories()
        
        processor = DataProcessor()
        X_train, X_test, y_train, train_ids, test_ids = processor.get_processed_data(
            use_feature_selection=True
        )
        
        from sklearn.model_selection import train_test_split
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, 
            test_size=Config.VALIDATION_SIZE, 
            random_state=Config.RANDOM_STATE,
            stratify=y_train
        )
        
        trainer = ModelTraining()
        models, best_model = trainer.train_all_models(
            X_train_split, y_train_split,
            X_val, y_val,
            use_optimization=True,
            model_list=['lightgbm', 'xgboost']
        )
        
        print("하이퍼파라미터 최적화 완료")
        
        return models, best_model
        
    except Exception as e:
        print(f"최적화 중 오류 발생: {e}")
        raise

def run_fast_mode():
    """빠른 실행 모드"""
    print("=" * 50)
    print("   빠른 실행 모드")
    print("=" * 50)
    
    try:
        Config.create_directories()
        
        processor = DataProcessor()
        X_train, X_test, y_train, train_ids, test_ids = processor.get_processed_data(
            use_feature_selection=True
        )
        
        from sklearn.model_selection import train_test_split
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, 
            test_size=Config.VALIDATION_SIZE, 
            random_state=Config.RANDOM_STATE,
            stratify=y_train
        )
        
        trainer = ModelTraining()
        
        # LightGBM만 사용
        trainer.train_lightgbm(X_train_split, y_train_split, X_val, y_val)
        trainer.cross_validation(X_train_split, y_train_split)
        
        predictor = Prediction(trainer.best_model)
        test_predictions = predictor.predict(X_test)
        submission_df = predictor.create_submission_file(test_ids)
        
        print(f"빠른 실행 완료: {Config.RESULT_FILE}")
        
        return trainer.models, trainer.best_model
        
    except Exception as e:
        print(f"빠른 실행 중 오류 발생: {e}")
        raise

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "predict":
            run_prediction_only()
        elif mode == "optimize":
            run_hyperparameter_optimization()
        elif mode == "fast":
            run_fast_mode()
        else:
            print("사용법:")
            print("  python main.py          # 전체 실행")
            print("  python main.py predict  # 예측만 실행")
            print("  python main.py optimize # 최적화 포함 실행")
            print("  python main.py fast     # 빠른 실행")
    else:
        main()
# main.py

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from config import Config
from data_processor import DataProcessor
from model_trainer import ModelTrainer
from predictor import Predictor
from utils import setup_logging, memory_usage_check

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("   스마트 장비 이상신호 감지 시스템")
    print("=" * 60)
    
    # 로깅 설정
    logger = setup_logging()
    logger.info("시스템 시작")
    
    # 디렉터리 생성
    Config.create_directories()
    
    # 메모리 사용량 체크
    memory_usage_check()
    
    try:
        # 1. 데이터 전처리
        print("\n" + "=" * 40)
        print("1단계: 데이터 전처리")
        print("=" * 40)
        
        processor = DataProcessor()
        X_train, X_test, y_train, train_ids, test_ids = processor.get_processed_data(
            use_feature_selection=True,
            use_pca=False
        )
        
        print(f"훈련 데이터 형태: {X_train.shape}")
        print(f"테스트 데이터 형태: {X_test.shape}")
        print(f"타겟 분포: {pd.Series(y_train).value_counts().sort_index().values}")
        
        # 2. 모델 훈련
        print("\n" + "=" * 40)
        print("2단계: 모델 훈련")
        print("=" * 40)
        
        trainer = ModelTrainer()
        
        # 검증 데이터 분할 (선택적)
        from sklearn.model_selection import train_test_split
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, 
            test_size=0.2, 
            random_state=Config.RANDOM_STATE,
            stratify=y_train
        )
        
        print(f"훈련 세트: {X_train_split.shape}")
        print(f"검증 세트: {X_val.shape}")
        
        # 모델 훈련 (하이퍼파라미터 최적화 사용 여부 선택)
        models, best_model = trainer.train_all_models(
            X_train_split, y_train_split, 
            X_val, y_val,
            use_optimization=False  # 빠른 실행을 위해 False로 설정
        )
        
        print(f"훈련된 모델 개수: {len(models)}")
        print(f"최고 성능 모델: {type(best_model).__name__}")
        
        # 검증 세트 평가
        predictor_val = Predictor(best_model)
        val_predictions = predictor_val.predict(X_val)
        val_score = predictor_val.validate_predictions(y_val)
        
        print(f"검증 세트 Macro F1 Score: {val_score:.4f}")
        
        # 3. 테스트 예측
        print("\n" + "=" * 40)
        print("3단계: 테스트 예측")
        print("=" * 40)
        
        predictor = Predictor(best_model)
        
        # 단일 모델 예측
        test_predictions = predictor.predict(X_test)
        
        # 앙상블 예측 (더 나은 성능을 위해)
        ensemble_predictions = predictor.predict_with_ensemble(
            {name: model for name, model in models.items() 
             if name not in ['ensemble', 'stacking']},
            X_test
        )
        
        # 예측 분포 분석
        predictor.analyze_prediction_distribution()
        
        # 4. 제출 파일 생성
        print("\n" + "=" * 40)
        print("4단계: 제출 파일 생성")
        print("=" * 40)
        
        submission_df = predictor.create_submission_file(test_ids)
        
        print(f"제출 파일 생성 완료: {Config.RESULT_FILE}")
        print(f"제출 파일 형태: {submission_df.shape}")
        
        # 5. 피처 중요도 분석
        print("\n" + "=" * 40)
        print("5단계: 모델 분석")
        print("=" * 40)
        
        feature_importance = trainer.feature_importance_analysis(
            best_model, 
            X_train.columns.tolist()
        )
        
        # 최종 메모리 사용량
        final_memory = memory_usage_check()
        
        print("\n" + "=" * 60)
        print("시스템 실행 완료")
        print("=" * 60)
        print(f"최종 예측 파일: {Config.RESULT_FILE}")
        print(f"최고 모델 파일: {Config.MODEL_FILE}")
        print(f"최종 메모리 사용량: {final_memory:.2f} MB")
        
        logger.info("시스템 정상 완료")
        
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
        # 데이터 전처리
        processor = DataProcessor()
        X_train, X_test, y_train, train_ids, test_ids = processor.get_processed_data(
            use_feature_selection=True,
            use_pca=False
        )
        
        # 훈련된 모델 로드
        predictor = Predictor()
        predictor.load_trained_model()
        
        # 예측 수행
        predictions = predictor.predict(X_test)
        
        # 제출 파일 생성
        submission_df = predictor.create_submission_file(test_ids)
        
        print(f"예측 완료: {Config.RESULT_FILE}")
        
    except Exception as e:
        print(f"예측 중 오류 발생: {e}")
        raise

def run_hyperparameter_optimization():
    """하이퍼파라미터 최적화 전용 모드"""
    print("=" * 50)
    print("   하이퍼파라미터 최적화 모드")
    print("=" * 50)
    
    try:
        # 데이터 전처리
        processor = DataProcessor()
        X_train, X_test, y_train, train_ids, test_ids = processor.get_processed_data()
        
        # 모델 훈련 (최적화 포함)
        trainer = ModelTrainer()
        models, best_model = trainer.train_all_models(
            X_train, y_train,
            use_optimization=True
        )
        
        print("하이퍼파라미터 최적화 완료")
        
    except Exception as e:
        print(f"최적화 중 오류 발생: {e}")
        raise

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "predict":
            run_prediction_only()
        elif mode == "optimize":
            run_hyperparameter_optimization()
        else:
            print("사용법:")
            print("  python main.py          # 전체 실행")
            print("  python main.py predict  # 예측만 실행")
            print("  python main.py optimize # 최적화 포함 실행")
    else:
        main()
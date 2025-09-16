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
    initial_memory = memory_usage_check()
    
    try:
        # 1. 데이터 전처리
        print("\n" + "=" * 50)
        print("1단계: 데이터 전처리")
        print("=" * 50)
        
        processor = DataProcessor()
        X_train, X_test, y_train, train_ids, test_ids = processor.get_processed_data(
            use_feature_selection=True,
            use_pca=False,
            use_svd=False
        )
        
        print(f"훈련 데이터 형태: {X_train.shape}")
        print(f"테스트 데이터 형태: {X_test.shape}")
        print(f"타겟 분포: {pd.Series(y_train).value_counts().sort_index().values}")
        
        # 2. 모델 훈련
        print("\n" + "=" * 50)
        print("2단계: 모델 훈련")
        print("=" * 50)
        
        trainer = ModelTrainer()
        
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
        
        # 모델 훈련
        models, best_model = trainer.train_all_models(
            X_train_split, y_train_split, 
            X_val, y_val,
            use_optimization=False,
            model_list=['lightgbm', 'xgboost', 'random_forest', 'extra_trees', 'gradient_boosting']
        )
        
        print(f"훈련된 모델 개수: {len(models)}")
        print(f"최고 성능 모델: {type(best_model).__name__}")
        
        # 검증 세트 평가
        predictor_val = Predictor(best_model)
        val_predictions = predictor_val.predict(X_val)
        val_metrics = predictor_val.validate_predictions(y_val)
        
        if val_metrics:
            print(f"검증 세트 Macro F1 Score: {val_metrics['macro_f1']:.4f}")
        
        # 3. 테스트 예측
        print("\n" + "=" * 50)
        print("3단계: 테스트 예측")
        print("=" * 50)
        
        predictor = Predictor(best_model)
        
        # 단일 모델 예측
        test_predictions = predictor.predict(X_test)
        
        # 앙상블 예측
        ensemble_models = {name: model for name, model in models.items() 
                         if name not in ['hard_voting', 'soft_voting', 'stacking'] and 'bagging' not in name}
        
        if len(ensemble_models) >= 2:
            ensemble_predictions = predictor.predict_with_ensemble(
                ensemble_models, X_test, method='weighted_average'
            )
            
            # 앙상블 결과를 최종 예측으로 사용
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
        # 디렉터리 생성
        Config.create_directories()
        
        # 데이터 전처리
        processor = DataProcessor()
        X_train, X_test, y_train, train_ids, test_ids = processor.get_processed_data(
            use_feature_selection=True,
            use_pca=False,
            use_svd=False
        )
        
        # 훈련된 모델 로드
        predictor = Predictor()
        predictor.load_trained_model()
        
        # 예측 수행
        predictions = predictor.predict(X_test)
        
        # 제출 파일 생성
        submission_df = predictor.create_submission_file(test_ids)
        
        print(f"예측 완료: {Config.RESULT_FILE}")
        
        return submission_df
        
    except Exception as e:
        print(f"예측 중 오류 발생: {e}")
        raise

def run_hyperparameter_optimization():
    """하이퍼파라미터 최적화 전용 모드"""
    print("=" * 50)
    print("   하이퍼파라미터 최적화 모드")
    print("=" * 50)
    
    try:
        # 디렉터리 생성
        Config.create_directories()
        
        # 데이터 전처리
        processor = DataProcessor()
        X_train, X_test, y_train, train_ids, test_ids = processor.get_processed_data(
            use_feature_selection=True,
            use_pca=False
        )
        
        # 검증 데이터 분할
        from sklearn.model_selection import train_test_split
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, 
            test_size=Config.VALIDATION_SIZE, 
            random_state=Config.RANDOM_STATE,
            stratify=y_train
        )
        
        # 모델 훈련 (최적화 포함)
        trainer = ModelTrainer()
        models, best_model = trainer.train_all_models(
            X_train_split, y_train_split,
            X_val, y_val,
            use_optimization=True,
            model_list=['lightgbm', 'xgboost', 'random_forest']
        )
        
        print("하이퍼파라미터 최적화 완료")
        
        return models, best_model
        
    except Exception as e:
        print(f"최적화 중 오류 발생: {e}")
        raise

def run_ensemble_experiment():
    """앙상블 실험 모드"""
    print("=" * 50)
    print("   앙상블 실험 모드")
    print("=" * 50)
    
    try:
        # 디렉터리 생성
        Config.create_directories()
        
        # 데이터 전처리
        processor = DataProcessor()
        X_train, X_test, y_train, train_ids, test_ids = processor.get_processed_data(
            use_feature_selection=True,
            use_pca=False
        )
        
        # 검증 데이터 분할
        from sklearn.model_selection import train_test_split
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, 
            test_size=Config.VALIDATION_SIZE, 
            random_state=Config.RANDOM_STATE,
            stratify=y_train
        )
        
        # 모델 훈련
        trainer = ModelTrainer()
        models, best_model = trainer.train_all_models(
            X_train_split, y_train_split,
            X_val, y_val,
            use_optimization=False,
            model_list=['lightgbm', 'xgboost', 'random_forest', 'extra_trees', 'gradient_boosting']
        )
        
        # 다양한 앙상블 방법 실험
        predictor = Predictor()
        
        ensemble_methods = ['weighted_average', 'majority_vote', 'rank_average']
        ensemble_results = {}
        
        for method in ensemble_methods:
            print(f"\n{method} 앙상블 실험 중...")
            
            # 기본 모델들만 사용
            base_models = {name: model for name, model in models.items() 
                         if name in ['lightgbm', 'xgboost', 'random_forest', 'extra_trees']}
            
            predictions = predictor.predict_with_ensemble(
                base_models, X_val, method=method
            )
            
            val_metrics = predictor.validate_predictions(y_val)
            ensemble_results[method] = val_metrics
            
            if val_metrics:
                print(f"{method} Macro F1: {val_metrics['macro_f1']:.4f}")
        
        # 최고 성능 앙상블 방법 선택
        best_ensemble_method = max(ensemble_results.keys(), 
                                 key=lambda x: ensemble_results[x]['macro_f1'] if ensemble_results[x] else 0)
        
        print(f"\n최고 앙상블 방법: {best_ensemble_method}")
        
        return ensemble_results, best_ensemble_method
        
    except Exception as e:
        print(f"앙상블 실험 중 오류 발생: {e}")
        raise

def run_feature_selection_experiment():
    """피처 선택 실험 모드"""
    print("=" * 50)
    print("   피처 선택 실험 모드")
    print("=" * 50)
    
    try:
        # 디렉터리 생성
        Config.create_directories()
        
        # 기본 데이터 로드
        processor = DataProcessor()
        train_df, test_df = processor.load_and_preprocess_data()
        X_train, X_test, y_train = processor.feature_engineering(train_df, test_df)
        X_train, X_test = processor.scale_features(X_train, X_test)
        
        # 다양한 피처 선택 방법 실험
        selection_methods = ['mutual_info', 'f_classif', 'random_forest', 'extra_trees']
        k_values = [20, 30, 40, 50]
        
        results = []
        
        for method in selection_methods:
            for k in k_values:
                print(f"\n{method} (k={k}) 실험 중...")
                
                try:
                    X_train_selected, X_test_selected = processor.select_features(
                        X_train, X_test, y_train, method=method, k=k
                    )
                    
                    # 간단한 모델로 성능 테스트
                    from sklearn.ensemble import RandomForestClassifier
                    from sklearn.model_selection import cross_val_score
                    from sklearn.metrics import make_scorer, f1_score
                    
                    model = RandomForestClassifier(n_estimators=100, random_state=Config.RANDOM_STATE)
                    scorer = make_scorer(f1_score, average='macro')
                    
                    cv_scores = cross_val_score(model, X_train_selected, y_train, cv=3, scoring=scorer)
                    mean_score = cv_scores.mean()
                    
                    results.append({
                        'method': method,
                        'k': k,
                        'cv_score': mean_score,
                        'n_features': X_train_selected.shape[1]
                    })
                    
                    print(f"CV Score: {mean_score:.4f}")
                    
                except Exception as e:
                    print(f"{method} (k={k}) 실패: {e}")
                    continue
        
        # 결과 정리
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('cv_score', ascending=False)
        
        print("\n=== 피처 선택 실험 결과 ===")
        print(results_df)
        
        best_result = results_df.iloc[0]
        print(f"\n최고 성능: {best_result['method']} (k={best_result['k']}) - {best_result['cv_score']:.4f}")
        
        return results_df
        
    except Exception as e:
        print(f"피처 선택 실험 중 오류 발생: {e}")
        raise

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "predict":
            run_prediction_only()
        elif mode == "optimize":
            run_hyperparameter_optimization()
        elif mode == "ensemble":
            run_ensemble_experiment()
        elif mode == "feature":
            run_feature_selection_experiment()
        else:
            print("사용법:")
            print("  python main.py          # 전체 실행")
            print("  python main.py predict  # 예측만 실행")
            print("  python main.py optimize # 최적화 포함 실행")
            print("  python main.py ensemble # 앙상블 실험")
            print("  python main.py feature  # 피처 선택 실험")
    else:
        main()
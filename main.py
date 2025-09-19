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
            use_pca=False,
            scaling_method='robust'
        )
        
        print(f"훈련 데이터 형태: {X_train.shape}")
        print(f"테스트 데이터 형태: {X_test.shape}")
        print(f"타겟 분포: {pd.Series(y_train).value_counts().sort_index().values}")
        
        # 데이터 크기에 따른 설정 조정
        adjustments = Config.adjust_for_data_size(len(X_train), X_train.shape[1])
        print(f"데이터 크기 기반 조정: {adjustments}")
        
        # 2. 검증 전략 정교화
        print("\n" + "=" * 50)
        print("2단계: 검증 전략 설정")
        print("=" * 50)
        
        # 다중 검증 전략 적용
        from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
        
        # 첫 번째 분할: 홀드아웃 검증용
        sss = StratifiedShuffleSplit(
            n_splits=1, 
            test_size=Config.VALIDATION_SIZE, 
            random_state=Config.RANDOM_STATE
        )
        
        train_idx, val_idx = next(sss.split(X_train, y_train))
        X_train_split = X_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        y_train_split = y_train.iloc[train_idx]
        y_val = y_train.iloc[val_idx]
        
        print(f"훈련 세트: {X_train_split.shape}")
        print(f"검증 세트: {X_val.shape}")
        
        # 검증 세트 분포 확인
        val_distribution = pd.Series(y_val).value_counts().sort_index()
        print(f"검증 세트 클래스 분포:")
        print(val_distribution.head(10))
        
        # 3. 모델 훈련
        print("\n" + "=" * 50)
        print("3단계: 모델 훈련")
        print("=" * 50)
        
        trainer = ModelTraining()
        
        # 확장 모드 설정 사용
        extended_config = Config.get_extended_mode_config()
        
        # 모델 훈련
        models, best_model = trainer.train_all_models(
            X_train_split, y_train_split, 
            X_val, y_val,
            use_optimization=True,
            model_list=extended_config['models']
        )
        
        print(f"훈련된 모델 개수: {len(models)}")
        print(f"최고 성능 모델: {type(best_model).__name__}")
        
        # 4. 다중 검증 수행
        print("\n" + "=" * 50)
        print("4단계: 다중 검증 수행")
        print("=" * 50)
        
        # 홀드아웃 검증
        val_predictor = Prediction(best_model)
        val_predictions = val_predictor.predict(X_val)
        val_metrics = val_predictor.validate_predictions(y_val)
        
        if val_metrics:
            print(f"홀드아웃 검증 Macro F1 Score: {val_metrics['macro_f1']:.4f}")
            holdout_score = val_metrics['macro_f1']
        else:
            holdout_score = 0.0
        
        # 교차 검증 결과와 홀드아웃 결과 비교
        if trainer.cv_scores:
            best_model_name = max(trainer.cv_scores.keys(), 
                                key=lambda x: trainer.cv_scores[x]['mean'])
            cv_score = trainer.cv_scores[best_model_name]['mean']
            
            print(f"교차 검증 점수: {cv_score:.4f}")
            print(f"홀드아웃 검증 점수: {holdout_score:.4f}")
            print(f"검증 차이: {abs(cv_score - holdout_score):.4f}")
            
            # 과적합 경고
            if abs(cv_score - holdout_score) > 0.02:
                print("경고: 교차 검증과 홀드아웃 검증 차이가 큼. 과적합 의심")
                # 보수적 모델 선택
                conservative_models = ['random_forest', 'ridge', 'logistic_regression']
                for model_name in conservative_models:
                    if model_name in models:
                        best_model = models[model_name]
                        print(f"보수적 모델로 변경: {model_name}")
                        break
        
        # 5. 앙상블 검증 및 선택
        print("\n" + "=" * 50)
        print("5단계: 앙상블 검증")
        print("=" * 50)
        
        # 성능 임계값 이상 모델들로 앙상블 구성
        good_models = {}
        for name, model in models.items():
            if name in trainer.cv_scores:
                score = trainer.cv_scores[name]['mean']
                if score >= Config.MIN_CV_SCORE:
                    good_models[name] = model
                    print(f"{name}: {score:.4f} (포함)")
                else:
                    print(f"{name}: {score:.4f} (제외)")
        
        print(f"앙상블 후보 모델: {len(good_models)}개")
        
        # 6. 테스트 예측
        print("\n" + "=" * 50)
        print("6단계: 테스트 예측")
        print("=" * 50)
        
        predictor = Prediction(best_model)
        
        # 앙상블 예측 수행
        if len(good_models) >= 2:
            print(f"앙상블 사용 모델: {list(good_models.keys())}")
            ensemble_predictions = predictor.predict_with_ensemble(good_models, X_test)
        else:
            # 단일 모델 예측
            test_predictions = predictor.predict(X_test)
        
        # 예측 분포 분석
        distribution_info = predictor.analyze_prediction_distribution()
        
        # 7. 제출 파일 생성
        print("\n" + "=" * 50)
        print("7단계: 제출 파일 생성")
        print("=" * 50)
        
        submission_df = predictor.create_submission_file(
            test_ids, 
            apply_balancing=True
        )
        
        print(f"제출 파일 생성 완료: {Config.RESULT_FILE}")
        print(f"제출 파일 형태: {submission_df.shape}")
        
        # 8. 성능 분석 및 리포트
        print("\n" + "=" * 50)
        print("8단계: 성능 분석")
        print("=" * 50)
        
        # 피처 중요도 분석
        feature_importance = trainer.feature_importance_analysis(
            best_model, 
            X_train.columns.tolist()
        )
        
        # CV 결과 출력
        if trainer.cv_scores:
            print("\n=== 교차 검증 결과 ===")
            sorted_scores = sorted(trainer.cv_scores.items(), 
                                 key=lambda x: x[1]['mean'], reverse=True)
            for model_name, scores in sorted_scores:
                print(f"{model_name}: {scores['mean']:.4f} (+/- {scores['std']*2:.4f})")
        
        # 성능 예측 및 권장사항
        print("\n=== 성능 예측 분석 ===")
        if holdout_score > 0:
            expected_performance = min(holdout_score, cv_score) * 0.98  # 보수적 추정
            print(f"예상 실제 성능: {expected_performance:.4f}")
            print(f"목표 성능까지: {0.80 - expected_performance:.4f}점 필요")
            
            if expected_performance < 0.75:
                print("권장사항: 피처 엔지니어링 재검토 및 모델 다양성 확대")
            elif expected_performance < 0.78:
                print("권장사항: 하이퍼파라미터 튜닝 및 앙상블 가중치 조정")
            else:
                print("권장사항: 확률 보정 및 세밀한 앙상블 튜닝")
        
        # 최종 메모리 사용량 및 요약
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
            print(f"홀드아웃 검증 점수: {val_metrics['macro_f1']:.4f}")
        
        logger.info("시스템 정상 완료")
        
        return {
            'models': models,
            'best_model': best_model,
            'val_metrics': val_metrics,
            'cv_scores': trainer.cv_scores,
            'distribution_info': distribution_info,
            'submission_df': submission_df,
            'expected_performance': expected_performance if 'expected_performance' in locals() else None
        }
        
    except Exception as e:
        logger.error(f"시스템 실행 중 오류 발생: {e}")
        print(f"오류 발생: {e}")
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
            scaling_method='standard'
        )
        
        # 검증 데이터 분할
        from sklearn.model_selection import train_test_split
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, 
            test_size=0.1,  # 빠른 모드에서는 검증 세트 축소
            random_state=Config.RANDOM_STATE,
            stratify=y_train
        )
        
        print(f"훈련 세트: {X_train_split.shape}")
        print(f"검증 세트: {X_val.shape}")
        
        # 빠른 모드 설정
        fast_config = Config.get_fast_mode_config()
        
        # 모델 훈련
        trainer = ModelTraining()
        models, best_model = trainer.train_all_models(
            X_train_split, y_train_split,
            X_val, y_val,
            use_optimization=False,
            model_list=fast_config['models']
        )
        
        print(f"훈련된 모델 개수: {len(models)}")
        print(f"최고 성능 모델: {type(best_model).__name__}")
        
        # 테스트 예측
        predictor = Prediction(best_model)
        test_predictions = predictor.predict(X_test)
        
        # 제출 파일 생성
        submission_df = predictor.create_submission_file(test_ids, apply_balancing=True)
        
        print(f"빠른 실행 완료: {Config.RESULT_FILE}")
        
        return models, best_model, submission_df
        
    except Exception as e:
        print(f"빠른 실행 중 오류 발생: {e}")
        raise

def run_prediction_only():
    """훈련된 모델로만 예측 수행"""
    print("=" * 50)
    print("   예측 전용 모드")
    print("=" * 50)
    
    try:
        Config.create_directories()
        
        # 데이터 전처리
        processor = DataProcessor()
        X_train, X_test, y_train, train_ids, test_ids = processor.get_processed_data(
            use_feature_selection=True
        )
        
        # 훈련된 모델 로드
        predictor = Prediction()
        predictor.load_trained_model()
        
        # 예측 수행
        predictions = predictor.predict(X_test)
        
        # 제출 파일 생성
        submission_df = predictor.create_submission_file(test_ids, apply_balancing=True)
        
        print(f"예측 완료: {Config.RESULT_FILE}")
        
        return submission_df
        
    except Exception as e:
        print(f"예측 중 오류 발생: {e}")
        raise

def run_performance_mode():
    """성능 중심 모드"""
    print("=" * 50)
    print("   성능 중심 모드")
    print("=" * 50)
    
    try:
        Config.create_directories()
        
        # 데이터 전처리 - 정밀도 우선
        processor = DataProcessor()
        X_train, X_test, y_train, train_ids, test_ids = processor.get_processed_data(
            use_feature_selection=True,
            use_pca=False,
            scaling_method='quantile'  # 분포 정규화
        )
        
        # 성능 중심 검증 데이터 분할
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=Config.RANDOM_STATE)
        train_idx, val_idx = next(skf.split(X_train, y_train))
        
        X_train_split = X_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        y_train_split = y_train.iloc[train_idx]
        y_val = y_train.iloc[val_idx]
        
        # 성능 중심 설정
        performance_config = Config.get_performance_config()
        
        # 모델 훈련
        trainer = ModelTraining()
        models, best_model = trainer.train_all_models(
            X_train_split, y_train_split,
            X_val, y_val,
            use_optimization=True,
            model_list=performance_config['models']
        )
        
        # 앙상블 예측
        predictor = Prediction()
        
        # 상위 성능 모델들만 선별
        top_models = {}
        if trainer.cv_scores:
            sorted_models = sorted(trainer.cv_scores.items(), 
                                 key=lambda x: x[1]['mean'], reverse=True)
            for name, score_info in sorted_models[:3]:  # 상위 3개만
                if name in models:
                    top_models[name] = models[name]
        
        if len(top_models) >= 2:
            ensemble_predictions = predictor.predict_with_ensemble(top_models, X_test)
        else:
            test_predictions = predictor.predict(X_test)
        
        # 제출 파일 생성
        submission_df = predictor.create_submission_file(test_ids, apply_balancing=True)
        
        print("성능 중심 실행 완료")
        print(f"제출 파일: {Config.RESULT_FILE}")
        
        return models, predictor, submission_df
        
    except Exception as e:
        print(f"성능 모드 실행 중 오류 발생: {e}")
        raise

def run_analysis_mode():
    """분석 전용 모드"""
    print("=" * 50)
    print("   분석 전용 모드")
    print("=" * 50)
    
    try:
        Config.create_directories()
        
        # 데이터 전처리
        processor = DataProcessor()
        X_train, X_test, y_train, train_ids, test_ids = processor.get_processed_data(
            use_feature_selection=False  # 모든 피처 유지
        )
        
        print("=== 데이터 분석 ===")
        print(f"원본 피처 수: {len(Config.FEATURE_COLUMNS)}")
        print(f"생성된 피처 수: {X_train.shape[1]}")
        print(f"피처 증가율: {(X_train.shape[1] / len(Config.FEATURE_COLUMNS) - 1) * 100:.1f}%")
        
        # 피처 상관관계 분석
        correlation_matrix = X_train.corr()
        high_corr = (correlation_matrix.abs() > 0.9).sum().sum() - X_train.shape[1]
        print(f"고상관 피처 쌍 수 (>0.9): {high_corr // 2}")
        
        # 클래스별 분포 분석
        class_stats = {}
        for class_id in range(Config.N_CLASSES):
            class_mask = y_train == class_id
            class_data = X_train[class_mask]
            class_stats[class_id] = {
                'count': len(class_data),
                'mean': class_data.mean().mean(),
                'std': class_data.std().mean()
            }
        
        print(f"\n=== 클래스별 통계 ===")
        for class_id, stats in class_stats.items():
            print(f"클래스 {class_id}: {stats['count']}개, "
                  f"평균: {stats['mean']:.4f}, 표준편차: {stats['std']:.4f}")
        
        return X_train, X_test, y_train, class_stats
        
    except Exception as e:
        print(f"분석 모드 실행 중 오류 발생: {e}")
        raise

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "fast":
            run_fast_mode()
        elif mode == "predict":
            run_prediction_only()
        elif mode == "performance":
            run_performance_mode()
        elif mode == "analysis":
            run_analysis_mode()
        else:
            print("사용법:")
            print("  python main.py              # 전체 실행")
            print("  python main.py fast         # 빠른 실행")
            print("  python main.py predict      # 예측만 실행")
            print("  python main.py performance  # 성능 중심 실행")
            print("  python main.py analysis     # 분석 전용 실행")
    else:
        main()
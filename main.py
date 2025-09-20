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
        
        # 2. 강화된 검증 전략 설정
        print("\n" + "=" * 50)
        print("2단계: 검증 전략 설정")
        print("=" * 50)
        
        from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
        
        # 다중 검증 전략 적용
        # 1) 시간 기반 검증 (첫 60%로 훈련, 다음 20%로 검증, 마지막 20%는 최종 테스트용)
        n_samples = len(X_train)
        train_end = int(n_samples * 0.6)
        val_end = int(n_samples * 0.8)
        
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
        
        # 2) 계층화 검증 추가
        from sklearn.model_selection import train_test_split
        X_train_strat, X_val_strat, y_train_strat, y_val_strat = train_test_split(
            X_train, y_train, 
            test_size=0.25,
            random_state=Config.RANDOM_STATE,
            stratify=y_train
        )
        
        print(f"\n계층화 분할:")
        print(f"  훈련 세트: {X_train_strat.shape}")
        print(f"  검증 세트: {X_val_strat.shape}")
        
        # 검증 세트 분포 확인
        val_distribution = pd.Series(y_val_time).value_counts().sort_index()
        print(f"\n시간 기반 검증 세트 클래스 분포 (상위 10개):")
        print(val_distribution.head(10))
        
        # 3. 모델 훈련 (과적합 방지 강화)
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
        
        # 4. 다중 검증 수행
        print("\n" + "=" * 50)
        print("4단계: 검증 수행")
        print("=" * 50)
        
        validation_scores = {}
        
        # 시간 기반 검증
        if best_model is not None:
            time_predictor = Prediction(best_model)
            time_predictions = time_predictor.predict(X_val_time)
            time_metrics = time_predictor.validate_predictions(y_val_time)
            
            if time_metrics:
                print(f"시간 기반 검증 Macro F1: {time_metrics['macro_f1']:.4f}")
                validation_scores['time_based'] = time_metrics['macro_f1']
        
        # 계층화 검증
        if best_model is not None:
            strat_predictor = Prediction(best_model)
            strat_predictions = strat_predictor.predict(X_val_strat)
            strat_metrics = strat_predictor.validate_predictions(y_val_strat)
            
            if strat_metrics:
                print(f"계층화 검증 Macro F1: {strat_metrics['macro_f1']:.4f}")
                validation_scores['stratified'] = strat_metrics['macro_f1']
        
        # 홀드아웃 검증 (최종 검증)
        if best_model is not None:
            holdout_predictor = Prediction(best_model)
            holdout_predictions = holdout_predictor.predict(X_holdout)
            holdout_metrics = holdout_predictor.validate_predictions(y_holdout)
            
            if holdout_metrics:
                print(f"홀드아웃 검증 Macro F1: {holdout_metrics['macro_f1']:.4f}")
                validation_scores['holdout'] = holdout_metrics['macro_f1']
        
        # 교차 검증 결과와 비교
        if trainer.cv_scores and validation_scores:
            best_model_name = max(trainer.cv_scores.keys(), 
                                key=lambda x: trainer.cv_scores[x]['mean'])
            cv_score = trainer.cv_scores[best_model_name]['mean']
            
            print(f"\n검증 점수 비교:")
            print(f"  교차 검증: {cv_score:.4f}")
            for val_type, score in validation_scores.items():
                print(f"  {val_type}: {score:.4f}")
                diff = abs(cv_score - score)
                print(f"    차이: {diff:.4f}")
                if diff > 0.05:
                    print(f"    경고: 과적합 위험 (차이 > 0.05)")
        
        # 5. 앙상블 예측 (보수적 접근)
        print("\n" + "=" * 50)
        print("5단계: 테스트 예측")
        print("=" * 50)
        
        if best_model is not None:
            predictor = Prediction(best_model)
            
            # 전체 훈련 데이터로 재훈련 (과적합 방지를 위해 정규화 강화)
            print("전체 훈련 데이터로 최종 모델 재훈련")
            
            # 보수적 파라미터로 재훈련
            conservative_trainer = ModelTraining()
            final_models, final_best_model = conservative_trainer.train_conservative_models(
                X_train, y_train
            )
            
            if final_best_model is not None:
                final_predictor = Prediction(final_best_model)
                test_predictions = final_predictor.predict(X_test)
                
                # 예측 분포 분석
                distribution_info = final_predictor.analyze_prediction_distribution()
            else:
                test_predictions = predictor.predict(X_test)
                distribution_info = predictor.analyze_prediction_distribution()
        
        # 6. 제출 파일 생성 (보수적 균형 조정)
        print("\n" + "=" * 50)
        print("6단계: 제출 파일 생성")
        print("=" * 50)
        
        if 'final_predictor' in locals():
            submission_df = final_predictor.create_submission_file(
                test_ids, 
                apply_balancing=True
            )
        elif best_model is not None:
            submission_df = predictor.create_submission_file(
                test_ids, 
                apply_balancing=True
            )
        else:
            print("경고: 유효한 모델이 없어 제출 파일을 생성할 수 없습니다.")
            submission_df = None
            
        if submission_df is not None:
            print(f"제출 파일 생성 완료: {Config.RESULT_FILE}")
            print(f"제출 파일 형태: {submission_df.shape}")
        
        # 7. 성능 분석 및 리포트
        print("\n" + "=" * 50)
        print("7단계: 성능 분석")
        print("=" * 50)
        
        # CV 결과 출력
        if trainer.cv_scores:
            print("\n교차 검증 결과")
            sorted_scores = sorted(trainer.cv_scores.items(), 
                                 key=lambda x: x[1]['mean'], reverse=True)
            for model_name, scores in sorted_scores:
                mean_score = scores['mean']
                std_score = scores['std']
                print(f"{model_name:20s}: {mean_score:.4f} (+/- {std_score*2:.4f})")
        
        # 검증 점수 안정성 분석
        if validation_scores:
            scores = list(validation_scores.values())
            mean_val = np.mean(scores)
            std_val = np.std(scores)
            
            print(f"\n검증 점수 안정성 분석:")
            print(f"  평균 검증 점수: {mean_val:.4f}")
            print(f"  검증 점수 표준편차: {std_val:.4f}")
            print(f"  안정성 지수: {1 - std_val:.4f}")
            
            if std_val > 0.02:
                print(f"  경고: 검증 점수 불안정 (std > 0.02)")
                print(f"  권장사항: 더 보수적인 모델 사용")
        
        # 성능 예측 및 권장사항
        print("\n성능 예측 분석")
        if validation_scores and trainer.cv_scores:
            cv_scores = [score['mean'] for score in trainer.cv_scores.values()]
            best_cv_score = max(cv_scores)
            
            # 보수적 성능 예측
            conservative_estimate = min(validation_scores.values()) * 0.95
            print(f"보수적 성능 예측: {conservative_estimate:.4f}")
            print(f"목표 성능 (0.80)까지: {0.80 - conservative_estimate:.4f}점 필요")
            
            if conservative_estimate < 0.70:
                print("권장사항: 피처 엔지니어링 재검토 및 데이터 품질 확인")
            elif conservative_estimate < 0.75:
                print("권장사항: 모델 복잡도 증가 및 앙상블 기법 강화")
            elif conservative_estimate < 0.78:
                print("권장사항: 하이퍼파라미터 세밀 조정")
            else:
                print("권장사항: 현재 접근 방법 유지 및 미세 조정")
        
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
        
        if validation_scores:
            for val_type, score in validation_scores.items():
                print(f"{val_type} 검증 점수: {score:.4f}")
        
        if trainer.cv_scores:
            best_cv_mean = max(score['mean'] for score in trainer.cv_scores.values())
            print(f"최고 교차 검증 점수: {best_cv_mean:.4f}")
        
        logger.info("시스템 정상 완료")
        
        # 반환값 구성
        result_dict = {
            'models': models,
            'best_model': best_model,
            'validation_scores': validation_scores,
            'cv_scores': trainer.cv_scores,
            'distribution_info': distribution_info if 'distribution_info' in locals() else None,
            'submission_df': submission_df,
            'conservative_estimate': conservative_estimate if 'conservative_estimate' in locals() else None
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
        
        # 데이터 전처리
        processor = DataProcessor()
        X_train, X_test, y_train, train_ids, test_ids = processor.get_processed_data(
            use_feature_selection=True,
            scaling_method='robust'
        )
        
        # 검증 데이터 분할
        from sklearn.model_selection import train_test_split
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, 
            test_size=0.2,
            random_state=Config.RANDOM_STATE,
            stratify=y_train
        )
        
        print(f"훈련 세트: {X_train_split.shape}")
        print(f"검증 세트: {X_val.shape}")
        
        # 모델 훈련
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
            
            # 제출 파일 생성
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
        
        if mode == "fast":
            run_fast_mode()
        else:
            print("사용법:")
            print("  python main.py         # 전체 실행")
            print("  python main.py fast    # 빠른 실행")
    else:
        main()
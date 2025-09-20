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
        
        # 2. 검증 전략 설정
        print("\n" + "=" * 50)
        print("2단계: 검증 전략 설정")
        print("=" * 50)
        
        from sklearn.model_selection import train_test_split
        
        # 시간 기반 검증을 위한 분할
        # 마지막 20%를 검증 데이터로 사용 (실제 운영 환경 시뮬레이션)
        split_idx = int(len(X_train) * 0.8)
        X_train_split = X_train.iloc[:split_idx]
        X_val = X_train.iloc[split_idx:]
        y_train_split = y_train.iloc[:split_idx]
        y_val = y_train.iloc[split_idx:]
        
        print(f"훈련 세트: {X_train_split.shape}")
        print(f"검증 세트: {X_val.shape}")
        
        # 검증 세트 분포 확인
        val_distribution = pd.Series(y_val).value_counts().sort_index()
        print(f"검증 세트 클래스 분포 (상위 10개):")
        print(val_distribution.head(10))
        
        # 3. 모델 훈련
        print("\n" + "=" * 50)
        print("3단계: 모델 훈련")
        print("=" * 50)
        
        trainer = ModelTraining()
        
        # 모델 훈련 (하이퍼파라미터 튜닝 포함)
        models, best_model = trainer.train_all_models(
            X_train_split, y_train_split, 
            X_val, y_val,
            use_optimization=True
        )
        
        print(f"훈련된 모델 개수: {len(models)}")
        if best_model is not None:
            print(f"최고 성능 모델: {type(best_model).__name__}")
        
        # 4. 검증 수행
        print("\n" + "=" * 50)
        print("4단계: 검증 수행")
        print("=" * 50)
        
        # 홀드아웃 검증
        if best_model is not None:
            val_predictor = Prediction(best_model)
            val_predictions = val_predictor.predict(X_val)
            val_metrics = val_predictor.validate_predictions(y_val)
            
            if val_metrics:
                print(f"홀드아웃 검증 Macro F1 Score: {val_metrics['macro_f1']:.4f}")
                holdout_score = val_metrics['macro_f1']
            else:
                holdout_score = 0.0
        else:
            holdout_score = 0.0
            val_metrics = None
        
        # 교차 검증 결과와 홀드아웃 결과 비교
        if trainer.cv_scores:
            best_model_name = max(trainer.cv_scores.keys(), 
                                key=lambda x: trainer.cv_scores[x]['mean'])
            cv_score = trainer.cv_scores[best_model_name]['mean']
            
            print(f"교차 검증 점수: {cv_score:.4f}")
            print(f"홀드아웃 검증 점수: {holdout_score:.4f}")
            
            if holdout_score > 0:
                print(f"검증 차이: {abs(cv_score - holdout_score):.4f}")
                
                # 과적합 경고
                if abs(cv_score - holdout_score) > 0.05:
                    print("경고: 교차 검증과 홀드아웃 검증 차이가 큼. 과적합 의심")
        
        # 5. 테스트 예측
        print("\n" + "=" * 50)
        print("5단계: 테스트 예측")
        print("=" * 50)
        
        if best_model is not None:
            predictor = Prediction(best_model)
            
            # 단일 모델 예측
            test_predictions = predictor.predict(X_test)
            
            # 예측 분포 분석
            distribution_info = predictor.analyze_prediction_distribution()
        
        # 6. 제출 파일 생성
        print("\n" + "=" * 50)
        print("6단계: 제출 파일 생성")
        print("=" * 50)
        
        if best_model is not None:
            submission_df = predictor.create_submission_file(
                test_ids, 
                apply_balancing=True
            )
            
            print(f"제출 파일 생성 완료: {Config.RESULT_FILE}")
            print(f"제출 파일 형태: {submission_df.shape}")
        else:
            print("경고: 유효한 모델이 없어 제출 파일을 생성할 수 없습니다.")
            submission_df = None
        
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
        
        # 성능 예측 및 권장사항
        print("\n성능 예측 분석")
        if holdout_score > 0 and trainer.cv_scores:
            best_cv_score = max(score['mean'] for score in trainer.cv_scores.values())
            expected_performance = min(holdout_score, best_cv_score) * 0.98
            print(f"예상 실제 성능: {expected_performance:.4f}")
            print(f"목표 성능 (0.80)까지: {0.80 - expected_performance:.4f}점 필요")
            
            if expected_performance < 0.72:
                print("권장사항: 피처 엔지니어링 재검토 필요")
            elif expected_performance < 0.75:
                print("권장사항: 하이퍼파라미터 튜닝 확대")
            elif expected_performance < 0.78:
                print("권장사항: 앙상블 기법 활용")
            else:
                print("권장사항: 현재 접근 방법 유지")
        
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
        
        if trainer.cv_scores:
            best_cv_mean = max(score['mean'] for score in trainer.cv_scores.values())
            print(f"최고 교차 검증 점수: {best_cv_mean:.4f}")
        
        logger.info("시스템 정상 완료")
        
        # 반환값 구성
        result_dict = {
            'models': models,
            'best_model': best_model,
            'val_metrics': val_metrics,
            'cv_scores': trainer.cv_scores,
            'distribution_info': distribution_info if 'distribution_info' in locals() else None,
            'submission_df': submission_df,
            'expected_performance': expected_performance if 'expected_performance' in locals() else None
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
# prediction.py

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

from config import Config
from utils import timer, load_model, calculate_macro_f1, validate_predictions, create_submission_template

class Prediction:
    def __init__(self, model=None):
        self.model = model
        self.predictions = None
        self.prediction_probabilities = None
        self.ensemble_models = {}
        
    def load_trained_model(self, model_path=None):
        """훈련된 모델 로드"""
        if model_path is None:
            model_path = Config.MODEL_FILE
            
        try:
            self.model = load_model(model_path)
            print(f"모델 로드 완료: {model_path}")
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            raise
    
    def load_ensemble_models(self, model_dict):
        """앙상블 모델들 로드"""
        self.ensemble_models = model_dict
        print(f"앙상블 모델 로드 완료: {len(self.ensemble_models)}개")
    
    @timer
    def predict(self, X_test):
        """단일 모델 예측"""
        if self.model is None:
            raise ValueError("모델이 로드되지 않았습니다.")
        
        print("=== 단일 모델 예측 수행 중 ===")
        
        # 예측 확률
        if hasattr(self.model, 'predict_proba'):
            self.prediction_probabilities = self.model.predict_proba(X_test)
            print(f"예측 확률 형태: {self.prediction_probabilities.shape}")
        
        # 예측 클래스
        self.predictions = self.model.predict(X_test)
        print(f"예측 결과 형태: {self.predictions.shape}")
        
        # 예측 결과 검증
        validate_predictions(self.predictions, Config.N_CLASSES)
        
        return self.predictions
    
    @timer
    def predict_with_ensemble(self, models_dict, X_test, weights=None):
        """앙상블 예측"""
        print("=== 앙상블 예측 수행 중 ===")
        print(f"사용 모델 수: {len(models_dict)}")
        
        if weights is None:
            weights = [1.0] * len(models_dict)
        
        if len(weights) != len(models_dict):
            print("가중치 개수가 모델 개수와 맞지 않아 균등 가중치 사용")
            weights = [1.0] * len(models_dict)
        
        all_probabilities = []
        
        for i, (name, model) in enumerate(models_dict.items()):
            print(f"{name} 예측 중...")
            
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_test)
                    all_probabilities.append(proba * weights[i])
                else:
                    print(f"{name}은 확률 예측을 지원하지 않아 건너뜀")
                    
            except Exception as e:
                print(f"{name} 예측 실패: {e}")
                continue
        
        if not all_probabilities:
            raise ValueError("유효한 예측이 없습니다.")
        
        # 가중 평균 확률
        ensemble_probabilities = np.average(all_probabilities, axis=0, weights=weights[:len(all_probabilities)])
        ensemble_predictions = np.argmax(ensemble_probabilities, axis=1)
        
        self.prediction_probabilities = ensemble_probabilities
        self.predictions = ensemble_predictions
        
        print(f"앙상블 예측 완료: {len(self.predictions)}개 샘플")
        
        # 예측 결과 검증
        validate_predictions(self.predictions, Config.N_CLASSES)
        
        return self.predictions
    
    def balance_predictions(self, target_distribution=None):
        """예측 분포 균형 조정"""
        if self.predictions is None or self.prediction_probabilities is None:
            print("예측이 수행되지 않아 균형 조정 불가")
            return self.predictions
        
        print("=== 예측 분포 균형 조정 시작 ===")
        
        # 현재 예측 분포 분석
        current_counts = np.bincount(self.predictions, minlength=Config.N_CLASSES)
        total_samples = len(self.predictions)
        
        # 목표 분포 설정 (균등 분포)
        if target_distribution is None:
            target_per_class = total_samples // Config.N_CLASSES
            target_distribution = np.full(Config.N_CLASSES, target_per_class)
            # 나머지 샘플 분배
            remainder = total_samples % Config.N_CLASSES
            target_distribution[:remainder] += 1
        
        print(f"목표 분포: 클래스당 약 {target_distribution[0]}개")
        
        # 확률 기반 재할당
        balanced_predictions = self.predictions.copy()
        
        for class_id in range(Config.N_CLASSES):
            current_count = current_counts[class_id]
            target_count = target_distribution[class_id]
            
            if current_count > target_count:
                # 과다 예측된 클래스에서 일부 제거
                class_indices = np.where(balanced_predictions == class_id)[0]
                class_probabilities = self.prediction_probabilities[class_indices, class_id]
                
                # 확률이 낮은 샘플들을 다른 클래스로 재할당
                remove_count = current_count - target_count
                low_prob_indices = class_indices[np.argsort(class_probabilities)[:remove_count]]
                
                # 두 번째로 높은 확률을 가진 클래스로 재할당
                for idx in low_prob_indices:
                    prob_sorted = np.argsort(self.prediction_probabilities[idx])[::-1]
                    second_best = prob_sorted[1]  # 두 번째로 높은 확률 클래스
                    balanced_predictions[idx] = second_best
        
        self.predictions = balanced_predictions
        
        # 균형 조정 결과 확인
        new_counts = np.bincount(self.predictions, minlength=Config.N_CLASSES)
        print("균형 조정 결과:")
        for i in range(Config.N_CLASSES):
            old_count = current_counts[i]
            new_count = new_counts[i]
            print(f"클래스 {i}: {old_count} → {new_count}")
        
        return self.predictions
    
    def analyze_prediction_distribution(self):
        """예측 분포 분석"""
        if self.predictions is None:
            print("예측이 수행되지 않았습니다.")
            return None
        
        print("=== 예측 분포 분석 ===")
        
        unique, counts = np.unique(self.predictions, return_counts=True)
        distribution = dict(zip(unique, counts))
        
        print("클래스별 예측 개수:")
        for class_id in range(Config.N_CLASSES):
            count = distribution.get(class_id, 0)
            percentage = (count / len(self.predictions)) * 100
            print(f"클래스 {class_id}: {count}개 ({percentage:.2f}%)")
        
        # 분포 균형도 분석
        total_predictions = len(self.predictions)
        expected_per_class = total_predictions / Config.N_CLASSES
        
        print(f"\n총 예측 개수: {total_predictions}")
        print(f"클래스당 기대 개수: {expected_per_class:.1f}")
        
        # 불균형 정도 계산
        imbalance_scores = []
        missing_classes = []
        
        for class_id in range(Config.N_CLASSES):
            count = distribution.get(class_id, 0)
            if count == 0:
                missing_classes.append(class_id)
                imbalance_scores.append(1.0)
            else:
                imbalance = abs(count - expected_per_class) / expected_per_class
                imbalance_scores.append(imbalance)
        
        avg_imbalance = np.mean(imbalance_scores)
        print(f"평균 불균형 정도: {avg_imbalance:.3f}")
        
        if missing_classes:
            print(f"경고: 예측되지 않은 클래스: {missing_classes}")
        
        # 불균형이 심한 경우 균형 조정 권장
        if avg_imbalance > 0.3:
            print("불균형이 심함. 균형 조정 권장")
        
        return {
            'distribution': distribution,
            'total_predictions': total_predictions,
            'expected_per_class': expected_per_class,
            'avg_imbalance': avg_imbalance,
            'missing_classes': missing_classes
        }
    
    @timer
    def create_submission_file(self, test_ids, output_path=None, predictions=None, apply_balancing=True):
        """제출 파일 생성"""
        if predictions is None:
            predictions = self.predictions
        
        if predictions is None:
            raise ValueError("예측이 수행되지 않았습니다.")
        
        if output_path is None:
            output_path = Config.RESULT_FILE
        
        print("=== 제출 파일 생성 중 ===")
        
        # 균형 조정 적용
        if apply_balancing and self.prediction_probabilities is not None:
            print("예측 분포 균형 조정 적용 중...")
            predictions = self.balance_predictions()
        
        # 기본 검증
        if len(test_ids) != len(predictions):
            raise ValueError(f"ID 개수({len(test_ids)})와 예측 개수({len(predictions)})가 일치하지 않습니다.")
        
        submission_df = create_submission_template(
            test_ids, predictions, 
            Config.ID_COLUMN, Config.TARGET_COLUMN
        )
        
        # 데이터 검증
        print(f"제출 파일 형태: {submission_df.shape}")
        print(f"ID 개수: {len(submission_df[Config.ID_COLUMN].unique())}")
        print(f"예측값 범위: {submission_df[Config.TARGET_COLUMN].min()} ~ {submission_df[Config.TARGET_COLUMN].max()}")
        
        # 중복 ID 확인
        if submission_df[Config.ID_COLUMN].duplicated().any():
            print("경고: 중복된 ID가 발견되었습니다.")
            duplicated_ids = submission_df[submission_df[Config.ID_COLUMN].duplicated()][Config.ID_COLUMN].values
            print(f"중복 ID: {duplicated_ids}")
        
        # 예측값 범위 확인
        invalid_predictions = (submission_df[Config.TARGET_COLUMN] < 0) | (submission_df[Config.TARGET_COLUMN] >= Config.N_CLASSES)
        if invalid_predictions.any():
            print("경고: 예측값이 유효한 범위를 벗어났습니다.")
            print(f"유효하지 않은 예측값: {submission_df[invalid_predictions][Config.TARGET_COLUMN].values}")
        
        # 파일 저장
        try:
            if isinstance(output_path, str):
                from pathlib import Path
                output_path = Path(output_path)
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            submission_df.to_csv(output_path, index=False)
            print(f"제출 파일 저장 완료: {output_path}")
        except Exception as e:
            print(f"파일 저장 실패: {e}")
            raise
        
        # 저장된 파일 검증
        try:
            saved_df = pd.read_csv(output_path)
            if saved_df.shape == submission_df.shape:
                print("파일 저장 검증 완료")
            else:
                print("경고: 저장된 파일 크기가 다릅니다.")
        except Exception as e:
            print(f"파일 검증 실패: {e}")
        
        return submission_df
    
    def validate_predictions(self, y_true=None):
        """예측 결과 검증"""
        if self.predictions is None:
            print("예측이 수행되지 않았습니다.")
            return None
        
        print("=== 예측 결과 검증 ===")
        
        # 기본 검증
        print(f"예측 개수: {len(self.predictions)}")
        print(f"고유 클래스 개수: {len(np.unique(self.predictions))}")
        print(f"예측값 범위: {self.predictions.min()} ~ {self.predictions.max()}")
        
        # 실제 레이블이 제공된 경우 성능 계산
        if y_true is not None:
            if len(y_true) != len(self.predictions):
                print("경고: 실제 레이블과 예측값의 개수가 다릅니다.")
                return None
            
            macro_f1 = calculate_macro_f1(y_true, self.predictions)
            print(f"Macro F1 Score: {macro_f1:.4f}")
            
            # 클래스별 성능 분석
            class_metrics = []
            
            for class_id in range(Config.N_CLASSES):
                class_mask = y_true == class_id
                if np.sum(class_mask) > 0:
                    class_pred = self.predictions[class_mask]
                    accuracy = np.mean(class_pred == class_id)
                    
                    # F1 스코어 계산
                    tp = np.sum((y_true == class_id) & (self.predictions == class_id))
                    fp = np.sum((y_true != class_id) & (self.predictions == class_id))
                    fn = np.sum((y_true == class_id) & (self.predictions != class_id))
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    
                    class_metrics.append({
                        'class': class_id,
                        'accuracy': accuracy,
                        'f1': f1,
                        'precision': precision,
                        'recall': recall
                    })
                    
                    print(f"클래스 {class_id} - 정확도: {accuracy:.4f}, F1: {f1:.4f}")
            
            avg_accuracy = np.mean([m['accuracy'] for m in class_metrics])
            avg_f1 = np.mean([m['f1'] for m in class_metrics])
            
            print(f"평균 클래스별 정확도: {avg_accuracy:.4f}")
            print(f"평균 클래스별 F1: {avg_f1:.4f}")
            
            return {
                'macro_f1': macro_f1,
                'avg_class_accuracy': avg_accuracy,
                'avg_class_f1': avg_f1,
                'class_metrics': class_metrics
            }
        
        return None
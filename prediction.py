# prediction.py

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from scipy import stats
from scipy.special import softmax
import warnings
warnings.filterwarnings('ignore')

from config import Config
from utils import timer, load_model, calculate_macro_f1, validate_predictions, create_submission_template

class Prediction:
    def __init__(self, model=None):
        self.model = model
        self.predictions = None
        self.prediction_probabilities = None
        self.prediction_history = []
        self.class_distributions = None
        
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
    
    @timer
    def predict(self, X_test, return_probabilities=True):
        """모델 예측"""
        if self.model is None:
            raise ValueError("모델이 로드되지 않았습니다.")
        
        print("모델 예측 수행 중")
        
        # 예측 확률
        if hasattr(self.model, 'predict_proba') and return_probabilities:
            self.prediction_probabilities = self.model.predict_proba(X_test)
            print(f"예측 확률 형태: {self.prediction_probabilities.shape}")
        elif hasattr(self.model, 'decision_function'):
            # SVM 등의 경우
            decision_scores = self.model.decision_function(X_test)
            self.prediction_probabilities = self._convert_to_probabilities(decision_scores)
            print(f"결정 함수 기반 확률 형태: {self.prediction_probabilities.shape}")
        
        # 예측 클래스
        self.predictions = self.model.predict(X_test)
        print(f"예측 결과 형태: {self.predictions.shape}")
        
        # 예측 결과 검증
        validate_predictions(self.predictions, Config.N_CLASSES)
        
        return self.predictions
    
    def _convert_to_probabilities(self, scores):
        """점수를 확률로 변환"""
        if scores.ndim == 1:
            # 이진 분류의 경우 다중 분류로 변환
            probs = np.zeros((len(scores), Config.N_CLASSES))
            # 임시로 균등 분포 할당
            probs.fill(1.0 / Config.N_CLASSES)
            return probs
        else:
            # 다중 분류의 경우 소프트맥스 적용
            return softmax(scores, axis=1)
    
    def balance_predictions(self, target_distribution=None, method='simple'):
        """예측 분포 균형 조정"""
        if self.predictions is None:
            print("예측이 수행되지 않아 균형 조정 불가")
            return self.predictions
        
        print(f"예측 분포 균형 조정 시작 ({method})")
        
        # 현재 예측 분포 분석
        current_counts = np.bincount(self.predictions, minlength=Config.N_CLASSES)
        total_samples = len(self.predictions)
        
        # 목표 분포 설정
        if target_distribution is None:
            # 균등 분포 목표
            base_count = total_samples // Config.N_CLASSES
            remainder = total_samples % Config.N_CLASSES
            
            target_distribution = np.full(Config.N_CLASSES, base_count)
            
            # 나머지를 랜덤하게 분배
            indices = np.random.RandomState(Config.RANDOM_STATE).choice(
                Config.N_CLASSES, remainder, replace=False
            )
            target_distribution[indices] += 1
        
        print(f"목표 분포: 평균 {np.mean(target_distribution):.0f}개")
        
        if method == 'probability_based' and self.prediction_probabilities is not None:
            balanced_predictions = self._probability_based_balancing(
                current_counts, target_distribution
            )
        else:
            balanced_predictions = self._simple_balancing(
                current_counts, target_distribution
            )
        
        self.predictions = balanced_predictions
        
        # 균형 조정 결과 확인
        new_counts = np.bincount(self.predictions, minlength=Config.N_CLASSES)
        print("균형 조정 결과:")
        for i in range(min(10, Config.N_CLASSES)):
            old_count = current_counts[i]
            new_count = new_counts[i]
            target_count = target_distribution[i]
            print(f"클래스 {i}: {old_count} → {new_count} (목표: {target_count})")
        
        return self.predictions
    
    def _probability_based_balancing(self, current_counts, target_distribution):
        """확률 기반 균형 조정"""
        balanced_predictions = self.predictions.copy()
        
        # 각 클래스별로 조정
        for class_id in range(Config.N_CLASSES):
            current_count = current_counts[class_id]
            target_count = target_distribution[class_id]
            
            if current_count > target_count:
                # 과다 예측된 클래스에서 일부 제거
                class_indices = np.where(balanced_predictions == class_id)[0]
                class_probabilities = self.prediction_probabilities[class_indices, class_id]
                
                remove_count = current_count - target_count
                low_prob_indices = class_indices[np.argsort(class_probabilities)[:remove_count]]
                
                # 다음으로 높은 확률을 가진 클래스로 재할당
                for idx in low_prob_indices:
                    prob_order = np.argsort(self.prediction_probabilities[idx])[::-1]
                    
                    # 현재 클래스를 제외한 다음 후보 찾기
                    for next_class in prob_order:
                        if next_class != class_id:
                            balanced_predictions[idx] = next_class
                            break
            
            elif current_count < target_count:
                # 부족한 클래스로 일부 재할당
                add_count = target_count - current_count
                
                # 다른 클래스 중에서 해당 클래스에 대한 확률이 높은 샘플들 찾기
                other_class_mask = balanced_predictions != class_id
                other_indices = np.where(other_class_mask)[0]
                
                if len(other_indices) > 0:
                    other_class_probs = self.prediction_probabilities[other_indices, class_id]
                    high_prob_indices = other_indices[np.argsort(other_class_probs)[-add_count:]]
                    
                    # 조건부 재할당
                    for idx in high_prob_indices:
                        original_class = balanced_predictions[idx]
                        original_prob = self.prediction_probabilities[idx, original_class]
                        target_prob = self.prediction_probabilities[idx, class_id]
                        
                        # 확률 차이가 크지 않은 경우에만 재할당
                        if target_prob > 0.15 and (original_prob - target_prob) < 0.4:
                            balanced_predictions[idx] = class_id
        
        return balanced_predictions
    
    def _simple_balancing(self, current_counts, target_distribution):
        """단순 균형 조정"""
        balanced_predictions = self.predictions.copy()
        
        # 과다 클래스에서 부족 클래스로 이동
        for class_id in range(Config.N_CLASSES):
            current_count = current_counts[class_id]
            target_count = target_distribution[class_id]
            
            if current_count > target_count:
                remove_count = current_count - target_count
                class_indices = np.where(balanced_predictions == class_id)[0]
                
                # 랜덤하게 일부 선택
                np.random.seed(Config.RANDOM_STATE + class_id)
                remove_indices = np.random.choice(class_indices, remove_count, replace=False)
                
                # 부족한 클래스들에 할당
                under_classes = [i for i in range(Config.N_CLASSES) 
                               if current_counts[i] < target_distribution[i]]
                
                if under_classes:
                    for idx in remove_indices:
                        target_class = np.random.choice(under_classes)
                        balanced_predictions[idx] = target_class
        
        return balanced_predictions
    
    def analyze_prediction_distribution(self):
        """예측 분포 분석"""
        if self.predictions is None:
            print("예측이 수행되지 않았습니다.")
            return None
        
        print("예측 분포 분석")
        
        unique, counts = np.unique(self.predictions, return_counts=True)
        distribution = dict(zip(unique, counts))
        
        print("클래스별 예측 개수:")
        total_predictions = len(self.predictions)
        for class_id in range(min(10, Config.N_CLASSES)):
            count = distribution.get(class_id, 0)
            percentage = (count / total_predictions) * 100
            print(f"클래스 {class_id:2d}: {count:4d}개 ({percentage:5.2f}%)")
        
        # 분포 통계
        expected_per_class = total_predictions / Config.N_CLASSES
        actual_counts = [distribution.get(i, 0) for i in range(Config.N_CLASSES)]
        
        print(f"\n총 예측 개수: {total_predictions}")
        print(f"클래스당 기대 개수: {expected_per_class:.1f}")
        print(f"실제 분포 표준편차: {np.std(actual_counts):.2f}")
        
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
            print(f"누락된 클래스: {missing_classes}")
        
        # 클래스 분포 저장
        self.class_distributions = distribution
        
        return {
            'distribution': distribution,
            'total_predictions': total_predictions,
            'expected_per_class': expected_per_class,
            'avg_imbalance': avg_imbalance,
            'missing_classes': missing_classes,
            'balance_status': 'good' if avg_imbalance < 0.2 else 'poor',
            'std': np.std(actual_counts)
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
        
        print("제출 파일 생성 중")
        
        # 균형 조정 적용
        if apply_balancing:
            print("예측 분포 균형 조정 적용 중")
            if self.prediction_probabilities is not None:
                predictions = self.balance_predictions(method='probability_based')
            else:
                predictions = self.balance_predictions(method='simple')
        
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
        
        # 예측값 범위 확인
        invalid_predictions = (submission_df[Config.TARGET_COLUMN] < 0) | (submission_df[Config.TARGET_COLUMN] >= Config.N_CLASSES)
        if invalid_predictions.any():
            print("경고: 예측값이 유효한 범위를 벗어났습니다.")
            invalid_count = invalid_predictions.sum()
            print(f"유효하지 않은 예측값 개수: {invalid_count}")
            
            # 유효하지 않은 값들을 가장 빈번한 클래스로 수정
            most_frequent_class = submission_df[Config.TARGET_COLUMN].mode()[0]
            submission_df.loc[invalid_predictions, Config.TARGET_COLUMN] = most_frequent_class
            print(f"유효하지 않은 예측값을 {most_frequent_class}으로 수정했습니다.")
        
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
                print(f"경고: 저장된 파일 크기가 다릅니다. 원본: {submission_df.shape}, 저장됨: {saved_df.shape}")
        except Exception as e:
            print(f"파일 검증 실패: {e}")
        
        # 예측 기록 저장
        self.prediction_history.append({
            'timestamp': pd.Timestamp.now(),
            'predictions': predictions.copy(),
            'distribution': np.bincount(predictions, minlength=Config.N_CLASSES),
            'method': 'balanced' if apply_balancing else 'raw'
        })
        
        return submission_df
    
    def validate_predictions(self, y_true=None):
        """예측 결과 검증"""
        if self.predictions is None:
            print("예측이 수행되지 않았습니다.")
            return None
        
        print("예측 결과 검증")
        
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
                        'recall': recall,
                        'support': np.sum(class_mask)
                    })
                    
                    if class_id < 10:  # 상위 10개 클래스만 출력
                        print(f"클래스 {class_id:2d} - 정확도: {accuracy:.4f}, F1: {f1:.4f}, 지원: {np.sum(class_mask):4d}")
            
            avg_accuracy = np.mean([m['accuracy'] for m in class_metrics])
            avg_f1 = np.mean([m['f1'] for m in class_metrics])
            
            print(f"평균 클래스별 정확도: {avg_accuracy:.4f}")
            print(f"평균 클래스별 F1: {avg_f1:.4f}")
            
            # 성능이 낮은 클래스 식별
            low_performance_classes = [m for m in class_metrics if m['f1'] < 0.5]
            if low_performance_classes:
                print(f"\n성능이 낮은 클래스 ({len(low_performance_classes)}개):")
                for m in low_performance_classes[:5]:  # 상위 5개만
                    print(f"  클래스 {m['class']:2d}: F1={m['f1']:.3f}, 지원={m['support']:4d}")
            
            return {
                'macro_f1': macro_f1,
                'avg_class_accuracy': avg_accuracy,
                'avg_class_f1': avg_f1,
                'class_metrics': class_metrics,
                'low_performance_classes': low_performance_classes
            }
        
        return None
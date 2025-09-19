# prediction.py

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.calibration import CalibratedClassifierCV
from scipy import stats
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
        self.prediction_history = []
        
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
    def predict(self, X_test, return_probabilities=True):
        """단일 모델 예측"""
        if self.model is None:
            raise ValueError("모델이 로드되지 않았습니다.")
        
        print("=== 단일 모델 예측 수행 중 ===")
        
        # 예측 확률
        if hasattr(self.model, 'predict_proba') and return_probabilities:
            self.prediction_probabilities = self.model.predict_proba(X_test)
            print(f"예측 확률 형태: {self.prediction_probabilities.shape}")
        elif hasattr(self.model, 'decision_function'):
            # SVM 등의 경우
            decision_scores = self.model.decision_function(X_test)
            # 소프트맥스 변환으로 확률 근사
            self.prediction_probabilities = self._softmax(decision_scores)
            print(f"결정 함수 기반 확률 형태: {self.prediction_probabilities.shape}")
        
        # 예측 클래스
        self.predictions = self.model.predict(X_test)
        print(f"예측 결과 형태: {self.predictions.shape}")
        
        # 예측 결과 검증
        validate_predictions(self.predictions, Config.N_CLASSES)
        
        return self.predictions
    
    def _softmax(self, scores):
        """소프트맥스 변환"""
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    @timer
    def predict_with_ensemble(self, models_dict, X_test, method='weighted_average', weights=None):
        """앙상블 예측 - 다양한 방법 지원"""
        print("=== 앙상블 예측 수행 중 ===")
        print(f"사용 모델 수: {len(models_dict)}")
        print(f"앙상블 방법: {method}")
        
        # 모델별 예측 수집
        model_predictions = {}
        model_probabilities = {}
        
        for name, model in models_dict.items():
            print(f"{name} 예측 중...")
            
            try:
                # 예측 수행
                pred = model.predict(X_test)
                model_predictions[name] = pred
                
                # 확률 수집
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(X_test)
                elif hasattr(model, 'decision_function'):
                    decision_scores = model.decision_function(X_test)
                    prob = self._softmax(decision_scores)
                else:
                    # 원핫 인코딩으로 확률 근사
                    prob = np.zeros((len(pred), Config.N_CLASSES))
                    for i, p in enumerate(pred):
                        prob[i, p] = 1.0
                
                model_probabilities[name] = prob
                    
            except Exception as e:
                print(f"{name} 예측 실패: {e}")
                continue
        
        if not model_probabilities:
            raise ValueError("유효한 예측이 없습니다.")
        
        # 가중치 설정
        if weights is None:
            weights = {name: 1.0 for name in model_probabilities.keys()}
        
        # 앙상블 방법별 처리
        if method == 'weighted_average':
            ensemble_probabilities = self._weighted_average_ensemble(
                model_probabilities, weights
            )
        elif method == 'rank_average':
            ensemble_probabilities = self._rank_average_ensemble(
                model_probabilities, weights
            )
        elif method == 'geometric_mean':
            ensemble_probabilities = self._geometric_mean_ensemble(
                model_probabilities, weights
            )
        elif method == 'majority_vote':
            ensemble_predictions = self._majority_vote_ensemble(
                model_predictions, weights
            )
            self.predictions = ensemble_predictions
            validate_predictions(self.predictions, Config.N_CLASSES)
            return self.predictions
        else:
            # 기본값: 가중 평균
            ensemble_probabilities = self._weighted_average_ensemble(
                model_probabilities, weights
            )
        
        # 최종 예측
        ensemble_predictions = np.argmax(ensemble_probabilities, axis=1)
        
        self.prediction_probabilities = ensemble_probabilities
        self.predictions = ensemble_predictions
        
        print(f"앙상블 예측 완료: {len(self.predictions)}개 샘플")
        
        # 예측 결과 검증
        validate_predictions(self.predictions, Config.N_CLASSES)
        
        return self.predictions
    
    def _weighted_average_ensemble(self, model_probabilities, weights):
        """가중 평균 앙상블"""
        weighted_probs = []
        total_weight = sum(weights.values())
        
        for name, prob in model_probabilities.items():
            weight = weights.get(name, 1.0) / total_weight
            weighted_probs.append(prob * weight)
        
        return np.sum(weighted_probs, axis=0)
    
    def _rank_average_ensemble(self, model_probabilities, weights):
        """랭크 평균 앙상블"""
        ranked_probs = []
        
        for name, prob in model_probabilities.items():
            # 각 샘플에 대해 확률을 랭크로 변환
            ranked_prob = np.zeros_like(prob)
            for i in range(prob.shape[0]):
                ranks = stats.rankdata(prob[i])
                ranked_prob[i] = ranks / ranks.sum()
            
            weight = weights.get(name, 1.0)
            ranked_probs.append(ranked_prob * weight)
        
        total_weight = sum(weights.values())
        return np.sum(ranked_probs, axis=0) / total_weight
    
    def _geometric_mean_ensemble(self, model_probabilities, weights):
        """기하 평균 앙상블"""
        log_probs = []
        
        for name, prob in model_probabilities.items():
            # 로그 확률 계산 (0 방지를 위해 작은 값 추가)
            log_prob = np.log(prob + 1e-10)
            weight = weights.get(name, 1.0)
            log_probs.append(log_prob * weight)
        
        total_weight = sum(weights.values())
        avg_log_prob = np.sum(log_probs, axis=0) / total_weight
        
        # 지수 변환으로 확률 복구
        ensemble_prob = np.exp(avg_log_prob)
        
        # 정규화
        return ensemble_prob / ensemble_prob.sum(axis=1, keepdims=True)
    
    def _majority_vote_ensemble(self, model_predictions, weights):
        """다수결 투표 앙상블"""
        n_samples = len(next(iter(model_predictions.values())))
        ensemble_predictions = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            votes = {}
            
            for name, pred in model_predictions.items():
                vote = pred[i]
                weight = weights.get(name, 1.0)
                votes[vote] = votes.get(vote, 0) + weight
            
            # 최다 득표 클래스 선택
            ensemble_predictions[i] = max(votes.keys(), key=votes.get)
        
        return ensemble_predictions
    
    @timer
    def calibrate_probabilities(self, X_val, y_val, method='isotonic'):
        """확률 보정"""
        if self.model is None:
            raise ValueError("모델이 로드되지 않았습니다.")
        
        print(f"=== 확률 보정 시작 ({method}) ===")
        
        try:
            # 보정된 모델 생성
            calibrated_model = CalibratedClassifierCV(
                base_estimator=self.model,
                method=method,
                cv='prefit'  # 이미 훈련된 모델 사용
            )
            
            calibrated_model.fit(X_val, y_val)
            self.model = calibrated_model
            
            print("확률 보정 완료")
            return calibrated_model
            
        except Exception as e:
            print(f"확률 보정 실패: {e}")
            return None
    
    def confidence_filtering(self, confidence_threshold=0.8):
        """신뢰도 기반 예측 필터링"""
        if self.prediction_probabilities is None:
            print("확률 정보가 없어 신뢰도 필터링을 수행할 수 없습니다.")
            return self.predictions
        
        print(f"=== 신뢰도 필터링 시작 (임계값: {confidence_threshold}) ===")
        
        max_probs = np.max(self.prediction_probabilities, axis=1)
        high_confidence_mask = max_probs >= confidence_threshold
        
        high_conf_count = np.sum(high_confidence_mask)
        low_conf_count = len(max_probs) - high_conf_count
        
        print(f"고신뢰도 예측: {high_conf_count}개 ({high_conf_count/len(max_probs)*100:.1f}%)")
        print(f"저신뢰도 예측: {low_conf_count}개 ({low_conf_count/len(max_probs)*100:.1f}%)")
        
        # 저신뢰도 예측에 대한 처리
        if low_conf_count > 0:
            # 클래스 빈도 기반으로 저신뢰도 예측 보정
            filtered_predictions = self.predictions.copy()
            
            # 전체 예측 분포 계산
            unique, counts = np.unique(self.predictions, return_counts=True)
            class_probs = counts / len(self.predictions)
            
            # 저신뢰도 예측을 확률 분포에 따라 재할당
            low_conf_indices = np.where(~high_confidence_mask)[0]
            np.random.seed(Config.RANDOM_STATE)
            
            for idx in low_conf_indices:
                # 두 번째로 높은 확률 클래스 고려
                prob_order = np.argsort(self.prediction_probabilities[idx])[::-1]
                top2_classes = prob_order[:2]
                
                # 더 높은 확률의 클래스 선택
                filtered_predictions[idx] = top2_classes[0]
            
            self.predictions = filtered_predictions
            print("저신뢰도 예측 보정 완료")
        
        return self.predictions
    
    def balance_predictions(self, target_distribution=None, method='probability_based'):
        """예측 분포 균형 조정"""
        if self.predictions is None:
            print("예측이 수행되지 않아 균형 조정 불가")
            return self.predictions
        
        print(f"=== 예측 분포 균형 조정 시작 ({method}) ===")
        
        # 현재 예측 분포 분석
        current_counts = np.bincount(self.predictions, minlength=Config.N_CLASSES)
        total_samples = len(self.predictions)
        
        # 목표 분포 설정
        if target_distribution is None:
            target_per_class = total_samples // Config.N_CLASSES
            target_distribution = np.full(Config.N_CLASSES, target_per_class)
            remainder = total_samples % Config.N_CLASSES
            target_distribution[:remainder] += 1
        
        print(f"목표 분포: 클래스당 약 {target_distribution[0]}개")
        
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
        for i in range(Config.N_CLASSES):
            old_count = current_counts[i]
            new_count = new_counts[i]
            print(f"클래스 {i}: {old_count} → {new_count}")
        
        return self.predictions
    
    def _probability_based_balancing(self, current_counts, target_distribution):
        """확률 기반 균형 조정"""
        balanced_predictions = self.predictions.copy()
        
        for class_id in range(Config.N_CLASSES):
            current_count = current_counts[class_id]
            target_count = target_distribution[class_id]
            
            if current_count > target_count:
                # 과다 예측된 클래스에서 일부 제거
                class_indices = np.where(balanced_predictions == class_id)[0]
                class_probabilities = self.prediction_probabilities[class_indices, class_id]
                
                remove_count = current_count - target_count
                low_prob_indices = class_indices[np.argsort(class_probabilities)[:remove_count]]
                
                # 두 번째로 높은 확률을 가진 클래스로 재할당
                for idx in low_prob_indices:
                    prob_sorted = np.argsort(self.prediction_probabilities[idx])[::-1]
                    second_best = prob_sorted[1]
                    balanced_predictions[idx] = second_best
            
            elif current_count < target_count:
                # 부족한 클래스로 일부 재할당
                add_count = target_count - current_count
                
                # 다른 클래스 중에서 해당 클래스에 대한 확률이 높은 샘플들 찾기
                other_class_mask = balanced_predictions != class_id
                other_indices = np.where(other_class_mask)[0]
                
                if len(other_indices) > 0:
                    other_class_probs = self.prediction_probabilities[other_indices, class_id]
                    high_prob_indices = other_indices[np.argsort(other_class_probs)[-add_count:]]
                    
                    # 조건부 재할당 (원래 예측과의 차이가 크지 않은 경우만)
                    for idx in high_prob_indices:
                        original_prob = self.prediction_probabilities[idx, balanced_predictions[idx]]
                        target_prob = self.prediction_probabilities[idx, class_id]
                        
                        if target_prob > 0.3 and (original_prob - target_prob) < 0.4:
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
                
                # 부족한 클래스 중 하나로 재할당
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
        
        # 성능 예측
        if avg_imbalance < 0.1:
            print("분포 상태: 매우 균형적")
        elif avg_imbalance < 0.2:
            print("분포 상태: 균형적")
        elif avg_imbalance < 0.4:
            print("분포 상태: 약간 불균형")
        else:
            print("분포 상태: 심각한 불균형 - 균형 조정 권장")
        
        return {
            'distribution': distribution,
            'total_predictions': total_predictions,
            'expected_per_class': expected_per_class,
            'avg_imbalance': avg_imbalance,
            'missing_classes': missing_classes,
            'balance_status': 'good' if avg_imbalance < 0.2 else 'poor'
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
        if apply_balancing:
            print("예측 분포 균형 조정 적용 중...")
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
            duplicated_ids = submission_df[submission_df[Config.ID_COLUMN].duplicated()][Config.ID_COLUMN].values
            print(f"중복 ID 개수: {len(duplicated_ids)}")
        
        # 예측값 범위 확인
        invalid_predictions = (submission_df[Config.TARGET_COLUMN] < 0) | (submission_df[Config.TARGET_COLUMN] >= Config.N_CLASSES)
        if invalid_predictions.any():
            print("경고: 예측값이 유효한 범위를 벗어났습니다.")
            invalid_count = invalid_predictions.sum()
            print(f"유효하지 않은 예측값 개수: {invalid_count}")
            
            # 유효하지 않은 값들을 0으로 수정
            submission_df.loc[invalid_predictions, Config.TARGET_COLUMN] = 0
            print("유효하지 않은 예측값을 0으로 수정했습니다.")
        
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
                        'recall': recall,
                        'support': np.sum(class_mask)
                    })
                    
                    print(f"클래스 {class_id} - 정확도: {accuracy:.4f}, F1: {f1:.4f}, 지원: {np.sum(class_mask)}")
            
            avg_accuracy = np.mean([m['accuracy'] for m in class_metrics])
            avg_f1 = np.mean([m['f1'] for m in class_metrics])
            
            print(f"평균 클래스별 정확도: {avg_accuracy:.4f}")
            print(f"평균 클래스별 F1: {avg_f1:.4f}")
            
            # 성능이 낮은 클래스 식별
            low_performance_classes = [m for m in class_metrics if m['f1'] < 0.6]
            if low_performance_classes:
                print(f"\n성능이 낮은 클래스 ({len(low_performance_classes)}개):")
                for m in low_performance_classes:
                    print(f"  클래스 {m['class']}: F1={m['f1']:.3f}, 지원={m['support']}")
            
            return {
                'macro_f1': macro_f1,
                'avg_class_accuracy': avg_accuracy,
                'avg_class_f1': avg_f1,
                'class_metrics': class_metrics,
                'low_performance_classes': low_performance_classes
            }
        
        return None
    
    def get_prediction_summary(self):
        """예측 요약 정보 반환"""
        if not self.prediction_history:
            return None
        
        latest_prediction = self.prediction_history[-1]
        
        summary = {
            'total_predictions': len(self.predictions) if self.predictions is not None else 0,
            'prediction_methods': len(self.prediction_history),
            'latest_distribution': latest_prediction['distribution'].tolist(),
            'latest_timestamp': latest_prediction['timestamp'],
            'has_probabilities': self.prediction_probabilities is not None,
            'balance_quality': self.analyze_prediction_distribution()
        }
        
        return summary
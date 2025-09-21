# prediction.py

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from scipy import stats
from scipy.special import softmax
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d
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
        self.calibration_data = None
        self.confidence_scores = None
        
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
            decision_scores = self.model.decision_function(X_test)
            self.prediction_probabilities = self._convert_to_probabilities(decision_scores)
            print(f"결정 함수 기반 확률 형태: {self.prediction_probabilities.shape}")
        
        # 예측 클래스
        self.predictions = self.model.predict(X_test)
        print(f"예측 결과 형태: {self.predictions.shape}")
        
        # 신뢰도 점수 계산
        if self.prediction_probabilities is not None:
            self.confidence_scores = np.max(self.prediction_probabilities, axis=1)
        
        # 예측 결과 검증
        validate_predictions(self.predictions, Config.N_CLASSES)
        
        return self.predictions
    
    def _convert_to_probabilities(self, scores):
        """점수를 확률로 변환"""
        if scores.ndim == 1:
            probs = np.zeros((len(scores), Config.N_CLASSES))
            probs.fill(1.0 / Config.N_CLASSES)
            return probs
        else:
            return softmax(scores, axis=1)
    
    def _calibrate_probabilities(self, probabilities, method='temperature'):
        """확률 보정"""
        if method == 'temperature' and self.calibration_data is not None:
            try:
                temperature = self.calibration_data.get('temperature', 1.0)
                calibrated_probs = softmax(np.log(probabilities + 1e-10) / temperature, axis=1)
                return calibrated_probs
            except Exception as e:
                print(f"확률 보정 실패: {e}")
                return probabilities
        
        return probabilities
    
    def _temporal_smoothing(self, predictions, window_size=5):
        """시간적 스무딩 적용"""
        if not Config.PREDICTION_SMOOTHING or len(predictions) < window_size:
            return predictions
        
        try:
            # 가우시안 필터를 사용한 스무딩
            smoothed = gaussian_filter1d(predictions.astype(float), sigma=window_size/4)
            smoothed_predictions = np.round(smoothed).astype(int)
            
            # 유효 범위로 클램핑
            smoothed_predictions = np.clip(smoothed_predictions, 0, Config.N_CLASSES - 1)
            
            return smoothed_predictions
            
        except Exception as e:
            print(f"시간적 스무딩 실패: {e}")
            return predictions
    
    def _confidence_based_adjustment(self, predictions, probabilities, threshold=None):
        """신뢰도 기반 예측 조정"""
        if probabilities is None:
            return predictions
        
        if threshold is None:
            threshold = Config.CONFIDENCE_THRESHOLD
        
        adjusted_predictions = predictions.copy()
        max_probs = np.max(probabilities, axis=1)
        low_confidence_mask = max_probs < threshold
        
        print(f"낮은 신뢰도 예측 ({threshold} 미만): {np.sum(low_confidence_mask)}개")
        
        if np.sum(low_confidence_mask) > 0:
            # 낮은 신뢰도 예측에 대해 보수적 접근
            low_conf_indices = np.where(low_confidence_mask)[0]
            
            for idx in low_conf_indices:
                probs = probabilities[idx]
                sorted_indices = np.argsort(probs)[::-1]
                
                # 상위 2개 클래스 간 차이가 작으면 더 안정적인 클래스 선택
                top1_prob = probs[sorted_indices[0]]
                top2_prob = probs[sorted_indices[1]]
                
                if top1_prob - top2_prob < 0.1:
                    # 클래스 균형을 고려한 선택
                    current_dist = np.bincount(adjusted_predictions, minlength=Config.N_CLASSES)
                    expected_count = len(predictions) / Config.N_CLASSES
                    
                    top1_class = sorted_indices[0]
                    top2_class = sorted_indices[1]
                    
                    if current_dist[top2_class] < current_dist[top1_class]:
                        adjusted_predictions[idx] = top2_class
        
        return adjusted_predictions
    
    def _smart_balance_predictions(self, target_distribution=None, method='entropy_based'):
        """스마트 예측 분포 균형 조정"""
        if self.predictions is None or self.prediction_probabilities is None:
            print("예측이 수행되지 않아 균형 조정 불가")
            return self.predictions
        
        print(f"스마트 예측 분포 균형 조정 시작 ({method})")
        
        current_counts = np.bincount(self.predictions, minlength=Config.N_CLASSES)
        total_samples = len(self.predictions)
        
        if target_distribution is None:
            # 실제 분포를 고려한 목표 분포 설정
            base_count = total_samples // Config.N_CLASSES
            remainder = total_samples % Config.N_CLASSES
            
            target_distribution = np.full(Config.N_CLASSES, base_count)
            
            # 나머지를 클래스 우선순위에 따라 분배
            priority_classes = list(range(Config.N_CLASSES))
            np.random.RandomState(Config.RANDOM_STATE).shuffle(priority_classes)
            
            for i in range(remainder):
                target_distribution[priority_classes[i]] += 1
        
        print(f"목표 분포: 평균 {np.mean(target_distribution):.0f}개")
        
        if method == 'entropy_based':
            balanced_predictions = self._entropy_based_balancing(current_counts, target_distribution)
        elif method == 'probability_threshold':
            balanced_predictions = self._probability_threshold_balancing(current_counts, target_distribution)
        else:
            balanced_predictions = self._optimization_based_balancing(current_counts, target_distribution)
        
        self.predictions = balanced_predictions
        
        # 균형 조정 결과 확인
        new_counts = np.bincount(self.predictions, minlength=Config.N_CLASSES)
        print("균형 조정 결과:")
        for i in range(min(10, Config.N_CLASSES)):
            old_count = current_counts[i]
            new_count = new_counts[i]
            target_count = target_distribution[i]
            change = "↑" if new_count > old_count else "↓" if new_count < old_count else "→"
            print(f"클래스 {i:2d}: {old_count:3d} {change} {new_count:3d} (목표: {target_count:3d})")
        
        if Config.N_CLASSES > 10:
            print(f"... (총 {Config.N_CLASSES}개 클래스)")
        
        return self.predictions
    
    def _entropy_based_balancing(self, current_counts, target_distribution):
        """엔트로피 기반 균형 조정"""
        balanced_predictions = self.predictions.copy()
        
        # 엔트로피 계산 (낮을수록 확실한 예측)
        entropies = []
        for i in range(len(self.prediction_probabilities)):
            probs = self.prediction_probabilities[i]
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            entropies.append(entropy)
        
        entropies = np.array(entropies)
        
        # 각 클래스별로 조정 수행
        for class_id in range(Config.N_CLASSES):
            current_count = current_counts[class_id]
            target_count = target_distribution[class_id]
            
            if current_count > target_count:
                # 과다 예측된 클래스에서 제거
                remove_count = current_count - target_count
                class_indices = np.where(balanced_predictions == class_id)[0]
                
                # 높은 엔트로피(불확실한) 예측 우선 제거
                class_entropies = entropies[class_indices]
                remove_indices = class_indices[np.argsort(class_entropies)[-remove_count:]]
                
                # 각 샘플을 가장 적합한 클래스로 재할당
                for idx in remove_indices:
                    probs = self.prediction_probabilities[idx].copy()
                    probs[class_id] = 0  # 현재 클래스 제외
                    
                    # 부족한 클래스들에 우선순위 부여
                    for j in range(Config.N_CLASSES):
                        if current_counts[j] < target_distribution[j]:
                            probs[j] *= 2  # 부족한 클래스에 가중치
                    
                    best_class = np.argmax(probs)
                    balanced_predictions[idx] = best_class
                    current_counts[class_id] -= 1
                    current_counts[best_class] += 1
            
            elif current_count < target_count:
                # 부족한 클래스로 추가
                add_count = target_count - current_count
                
                # 다른 클래스에서 이 클래스로 이동할 후보 찾기
                other_indices = np.where(balanced_predictions != class_id)[0]
                
                if len(other_indices) > 0:
                    # 이 클래스에 대한 확률이 높은 순서대로 후보 선택
                    other_probs = self.prediction_probabilities[other_indices, class_id]
                    candidate_indices = other_indices[np.argsort(other_probs)[-add_count*2:]]
                    
                    added = 0
                    for idx in reversed(candidate_indices):
                        if added >= add_count:
                            break
                        
                        original_class = balanced_predictions[idx]
                        original_prob = self.prediction_probabilities[idx, original_class]
                        target_prob = self.prediction_probabilities[idx, class_id]
                        
                        # 확률 차이가 크지 않고, 원래 클래스가 과다한 경우에만 변경
                        if (target_prob > 0.15 and 
                            (original_prob - target_prob) < 0.4 and
                            current_counts[original_class] > target_distribution[original_class]):
                            
                            balanced_predictions[idx] = class_id
                            current_counts[original_class] -= 1
                            current_counts[class_id] += 1
                            added += 1
        
        return balanced_predictions
    
    def _probability_threshold_balancing(self, current_counts, target_distribution):
        """확률 임계값 기반 균형 조정"""
        balanced_predictions = self.predictions.copy()
        
        # 동적 임계값 계산
        prob_thresholds = {}
        for class_id in range(Config.N_CLASSES):
            class_probs = self.prediction_probabilities[:, class_id]
            prob_thresholds[class_id] = np.percentile(class_probs, 70)
        
        # 각 클래스별 조정
        for class_id in range(Config.N_CLASSES):
            current_count = current_counts[class_id]
            target_count = target_distribution[class_id]
            threshold = prob_thresholds[class_id]
            
            if current_count > target_count:
                remove_count = current_count - target_count
                class_indices = np.where(balanced_predictions == class_id)[0]
                
                # 임계값 이하의 확률을 가진 예측 우선 제거
                class_probs = self.prediction_probabilities[class_indices, class_id]
                low_prob_mask = class_probs < threshold
                low_prob_indices = class_indices[low_prob_mask]
                
                if len(low_prob_indices) >= remove_count:
                    remove_indices = low_prob_indices[:remove_count]
                else:
                    # 부족하면 확률이 낮은 순서로 추가 선택
                    remaining_remove = remove_count - len(low_prob_indices)
                    high_prob_indices = class_indices[~low_prob_mask]
                    
                    if len(high_prob_indices) > 0:
                        high_probs = self.prediction_probabilities[high_prob_indices, class_id]
                        additional_remove = high_prob_indices[np.argsort(high_probs)[:remaining_remove]]
                        remove_indices = np.concatenate([low_prob_indices, additional_remove])
                    else:
                        remove_indices = low_prob_indices
                
                # 재할당
                for idx in remove_indices:
                    prob_order = np.argsort(self.prediction_probabilities[idx])[::-1]
                    for next_class in prob_order:
                        if (next_class != class_id and 
                            current_counts[next_class] < target_distribution[next_class]):
                            balanced_predictions[idx] = next_class
                            current_counts[class_id] -= 1
                            current_counts[next_class] += 1
                            break
                    else:
                        # 적절한 클래스를 찾지 못한 경우 가장 확률이 높은 다른 클래스로
                        next_class = prob_order[1] if prob_order[1] != class_id else prob_order[0]
                        balanced_predictions[idx] = next_class
                        current_counts[class_id] -= 1
                        current_counts[next_class] += 1
        
        return balanced_predictions
    
    def _optimization_based_balancing(self, current_counts, target_distribution):
        """최적화 기반 균형 조정"""
        balanced_predictions = self.predictions.copy()
        
        # 각 샘플의 클래스 변경 비용 계산
        n_samples = len(self.predictions)
        change_costs = np.zeros((n_samples, Config.N_CLASSES))
        
        for i in range(n_samples):
            original_class = self.predictions[i]
            original_prob = self.prediction_probabilities[i, original_class]
            
            for j in range(Config.N_CLASSES):
                if j == original_class:
                    change_costs[i, j] = 0
                else:
                    target_prob = self.prediction_probabilities[i, j]
                    change_costs[i, j] = original_prob - target_prob
        
        # 각 클래스별로 조정 수행
        for class_id in range(Config.N_CLASSES):
            current_count = current_counts[class_id]
            target_count = target_distribution[class_id]
            
            if current_count > target_count:
                remove_count = current_count - target_count
                class_indices = np.where(balanced_predictions == class_id)[0]
                
                # 제거할 샘플들을 선택 (확률이 낮은 순서대로)
                class_probs = self.prediction_probabilities[class_indices, class_id]
                remove_indices = class_indices[np.argsort(class_probs)[:remove_count]]
                
                # 각 샘플을 가장 적합한 클래스로 재할당
                for idx in remove_indices:
                    costs = change_costs[idx].copy()
                    costs[class_id] = np.inf  # 현재 클래스는 제외
                    
                    # 부족한 클래스들에 우선순위 부여
                    for j in range(Config.N_CLASSES):
                        if current_counts[j] < target_distribution[j]:
                            costs[j] *= 0.5  # 부족한 클래스에 보너스
                    
                    best_class = np.argmin(costs)
                    balanced_predictions[idx] = best_class
                    current_counts[class_id] -= 1
                    current_counts[best_class] += 1
            
            elif current_count < target_count:
                add_count = target_count - current_count
                
                # 다른 클래스에서 이 클래스로 이동할 후보 찾기
                other_indices = np.where(balanced_predictions != class_id)[0]
                
                if len(other_indices) > 0:
                    # 이 클래스에 대한 확률이 높은 순서대로 후보 선택
                    other_probs = self.prediction_probabilities[other_indices, class_id]
                    candidate_indices = other_indices[np.argsort(other_probs)[-add_count*2:]]
                    
                    added = 0
                    for idx in reversed(candidate_indices):
                        if added >= add_count:
                            break
                        
                        original_class = balanced_predictions[idx]
                        cost = change_costs[idx, class_id]
                        
                        # 비용이 낮고, 원래 클래스가 과다한 경우에만 변경
                        if (cost < 0.5 and 
                            current_counts[original_class] > target_distribution[original_class]):
                            
                            balanced_predictions[idx] = class_id
                            current_counts[original_class] -= 1
                            current_counts[class_id] += 1
                            added += 1
        
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
        
        if Config.N_CLASSES > 10:
            print(f"... (총 {Config.N_CLASSES}개 클래스)")
        
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
            'balance_status': 'good' if avg_imbalance < 0.2 else 'moderate' if avg_imbalance < 0.4 else 'poor',
            'std': np.std(actual_counts),
            'balance_score': 1 - avg_imbalance
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
        
        # 신뢰도 기반 조정
        if self.prediction_probabilities is not None:
            predictions = self._confidence_based_adjustment(predictions, self.prediction_probabilities)
        
        # 시간적 스무딩
        predictions = self._temporal_smoothing(predictions)
        
        # 균형 조정 적용
        if apply_balancing:
            print("예측 분포 균형 조정 적용 중")
            if self.prediction_probabilities is not None:
                predictions = self._smart_balance_predictions(method='entropy_based')
            else:
                predictions = self._smart_balance_predictions(method='optimization')
        
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
        distribution_info = self.analyze_prediction_distribution()
        
        self.prediction_history.append({
            'timestamp': pd.Timestamp.now(),
            'predictions': predictions.copy(),
            'distribution': np.bincount(predictions, minlength=Config.N_CLASSES),
            'method': 'balanced' if apply_balancing else 'raw',
            'balance_score': distribution_info['balance_score'] if distribution_info else 0.0
        })
        
        return submission_df
    
    def validate_predictions(self, y_true=None):
        """예측 결과 검증"""
        if self.predictions is None:
            print("예측이 수행되지 않았습니다.")
            return None
        
        print("예측 결과 검증")
        
        print(f"예측 개수: {len(self.predictions)}")
        print(f"고유 클래스 개수: {len(np.unique(self.predictions))}")
        print(f"예측값 범위: {self.predictions.min()} ~ {self.predictions.max()}")
        
        # 신뢰도 분석
        if self.confidence_scores is not None:
            print(f"평균 신뢰도: {np.mean(self.confidence_scores):.4f}")
            print(f"신뢰도 표준편차: {np.std(self.confidence_scores):.4f}")
            
            # 신뢰도 구간별 분포
            high_conf = np.sum(self.confidence_scores > 0.8)
            medium_conf = np.sum((self.confidence_scores > 0.5) & (self.confidence_scores <= 0.8))
            low_conf = np.sum(self.confidence_scores <= 0.5)
            
            print(f"높은 신뢰도 (>0.8): {high_conf}개 ({high_conf/len(self.predictions)*100:.1f}%)")
            print(f"중간 신뢰도 (0.5-0.8): {medium_conf}개 ({medium_conf/len(self.predictions)*100:.1f}%)")
            print(f"낮은 신뢰도 (≤0.5): {low_conf}개 ({low_conf/len(self.predictions)*100:.1f}%)")
        
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
                        'f1_score': f1,
                        'precision': precision,
                        'recall': recall,
                        'support': np.sum(class_mask)
                    })
                    
                    if class_id < 10:
                        print(f"클래스 {class_id:2d} - 정확도: {accuracy:.4f}, F1: {f1:.4f}, 지원: {np.sum(class_mask):4d}")
            
            avg_accuracy = np.mean([m['accuracy'] for m in class_metrics])
            avg_f1 = np.mean([m['f1'] for m in class_metrics])
            
            print(f"평균 클래스별 정확도: {avg_accuracy:.4f}")
            print(f"평균 클래스별 F1: {avg_f1:.4f}")
            
            # 성능이 낮은 클래스 식별
            low_performance_classes = [m for m in class_metrics if m['f1'] < Config.CLASS_PERFORMANCE_THRESHOLD]
            if low_performance_classes:
                print(f"\n성능이 낮은 클래스 ({len(low_performance_classes)}개):")
                for m in low_performance_classes[:5]:
                    print(f"  클래스 {m['class']:2d}: F1={m['f1']:.3f}, 지원={m['support']:4d}")
                
                if len(low_performance_classes) > 5:
                    print(f"  ... (총 {len(low_performance_classes)}개)")
            
            return {
                'macro_f1': macro_f1,
                'avg_class_accuracy': avg_accuracy,
                'avg_class_f1': avg_f1,
                'class_metrics': class_metrics,
                'low_performance_classes': low_performance_classes,
                'prediction_distribution': self.analyze_prediction_distribution(),
                'confidence_stats': {
                    'mean': np.mean(self.confidence_scores) if self.confidence_scores is not None else None,
                    'std': np.std(self.confidence_scores) if self.confidence_scores is not None else None,
                    'high_conf_ratio': high_conf/len(self.predictions) if self.confidence_scores is not None else None
                }
            }
        
        return None
    
    def get_prediction_confidence(self):
        """예측 신뢰도 분석"""
        if self.prediction_probabilities is None:
            print("예측 확률이 없어 신뢰도 분석 불가")
            return None
        
        max_probs = np.max(self.prediction_probabilities, axis=1)
        
        confidence_stats = {
            'mean_confidence': np.mean(max_probs),
            'median_confidence': np.median(max_probs),
            'std_confidence': np.std(max_probs),
            'min_confidence': np.min(max_probs),
            'max_confidence': np.max(max_probs)
        }
        
        # 신뢰도 구간별 분포
        confidence_bins = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
        bin_counts = np.histogram(max_probs, bins=confidence_bins)[0]
        
        confidence_distribution = {}
        for i in range(len(confidence_bins)-1):
            bin_name = f"{confidence_bins[i]:.1f}-{confidence_bins[i+1]:.1f}"
            confidence_distribution[bin_name] = bin_counts[i]
        
        print("예측 신뢰도 분석:")
        print(f"  평균 신뢰도: {confidence_stats['mean_confidence']:.4f}")
        print(f"  중앙값 신뢰도: {confidence_stats['median_confidence']:.4f}")
        print(f"  신뢰도 표준편차: {confidence_stats['std_confidence']:.4f}")
        
        print("\n신뢰도 구간별 분포:")
        for bin_name, count in confidence_distribution.items():
            percentage = (count / len(max_probs)) * 100
            print(f"  {bin_name}: {count:4d}개 ({percentage:5.1f}%)")
        
        return {
            'confidence_stats': confidence_stats,
            'confidence_distribution': confidence_distribution,
            'confidence_scores': max_probs
        }
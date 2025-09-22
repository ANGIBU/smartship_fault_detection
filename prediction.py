# prediction.py

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
from scipy import stats
from scipy.special import softmax
import warnings
warnings.filterwarnings('ignore')

from config import Config
from utils import timer, load_model, calculate_macro_f1, validate_predictions, create_submission_template

class PredictionProcessor:
    def __init__(self, model=None):
        self.model = model
        self.predictions = None
        self.prediction_probabilities = None
        self.confidence_scores = None
        self.calibrated_model = None
        
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
    
    def calibrate_model(self, X_val, y_val):
        """모델 확률 보정"""
        if self.model is None:
            print("모델이 로드되지 않았습니다")
            return
        
        try:
            print("모델 확률 보정 시작")
            self.calibrated_model = CalibratedClassifierCV(
                self.model,
                method='isotonic',
                cv=3
            )
            self.calibrated_model.fit(X_val, y_val)
            print("모델 확률 보정 완료")
        except Exception as e:
            print(f"모델 보정 실패: {e}")
            self.calibrated_model = None
    
    @timer
    def predict(self, X_test, use_calibrated=True):
        """모델 예측"""
        if self.model is None:
            raise ValueError("모델이 로드되지 않았습니다")
        
        print("모델 예측 수행 중")
        
        # 사용할 모델 선택
        active_model = self.calibrated_model if (use_calibrated and self.calibrated_model is not None) else self.model
        
        # 예측 확률
        if hasattr(active_model, 'predict_proba'):
            self.prediction_probabilities = active_model.predict_proba(X_test)
            print(f"예측 확률 형태: {self.prediction_probabilities.shape}")
        else:
            print("확률 예측 불가능")
            self.prediction_probabilities = None
        
        # 예측 클래스
        self.predictions = active_model.predict(X_test)
        print(f"예측 결과 형태: {self.predictions.shape}")
        
        # 신뢰도 점수 계산
        if self.prediction_probabilities is not None:
            self.confidence_scores = np.max(self.prediction_probabilities, axis=1)
        
        # 예측 결과 검증
        validate_predictions(self.predictions, Config.N_CLASSES)
        
        return self.predictions
    
    def _adjust_predictions_by_confidence(self, predictions, probabilities, threshold=0.6):
        """신뢰도 기반 예측 조정"""
        if probabilities is None:
            return predictions
        
        adjusted_predictions = predictions.copy()
        max_probs = np.max(probabilities, axis=1)
        low_confidence_mask = max_probs < threshold
        
        print(f"낮은 신뢰도 예측 ({threshold} 미만): {np.sum(low_confidence_mask)}개")
        
        if np.sum(low_confidence_mask) > 0:
            # 낮은 신뢰도 예측에 대해 상위 2개 클래스 고려
            low_conf_indices = np.where(low_confidence_mask)[0]
            
            for idx in low_conf_indices:
                probs = probabilities[idx]
                sorted_indices = np.argsort(probs)[::-1]
                
                # 상위 2개 클래스의 확률 차이가 작으면 더 균형잡힌 클래스 선택
                top1_prob = probs[sorted_indices[0]]
                top2_prob = probs[sorted_indices[1]]
                
                if top1_prob - top2_prob < 0.1:
                    # 현재 예측 분포를 고려하여 선택
                    current_pred_counts = np.bincount(adjusted_predictions, minlength=Config.N_CLASSES)
                    expected_count = len(predictions) / Config.N_CLASSES
                    
                    top1_class = sorted_indices[0]
                    top2_class = sorted_indices[1]
                    
                    # 예측이 적은 클래스를 선택
                    if current_pred_counts[top2_class] < current_pred_counts[top1_class]:
                        adjusted_predictions[idx] = top2_class
        
        return adjusted_predictions
    
    def _balance_prediction_distribution(self, predictions, probabilities=None, method='entropy'):
        """예측 분포 균형 조정"""
        if probabilities is None:
            print("확률 정보가 없어 분포 조정 불가")
            return predictions
        
        print(f"예측 분포 균형 조정 시작 ({method})")
        
        current_counts = np.bincount(predictions, minlength=Config.N_CLASSES)
        total_samples = len(predictions)
        expected_count = total_samples / Config.N_CLASSES
        
        # 목표 분포 설정
        target_counts = np.full(Config.N_CLASSES, int(expected_count))
        remainder = total_samples - target_counts.sum()
        
        # 나머지를 무작위로 분배
        if remainder > 0:
            random_classes = np.random.RandomState(Config.RANDOM_STATE).choice(
                Config.N_CLASSES, remainder, replace=False
            )
            for cls in random_classes:
                target_counts[cls] += 1
        
        balanced_predictions = predictions.copy()
        
        if method == 'entropy':
            # 엔트로피 기반 조정
            entropies = []
            for i in range(len(probabilities)):
                probs = probabilities[i]
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                entropies.append(entropy)
            
            entropies = np.array(entropies)
            
            # 각 클래스별로 조정
            for class_id in range(Config.N_CLASSES):
                current_count = current_counts[class_id]
                target_count = target_counts[class_id]
                
                if current_count > target_count:
                    # 과다 예측된 클래스에서 제거
                    remove_count = current_count - target_count
                    class_indices = np.where(balanced_predictions == class_id)[0]
                    
                    # 높은 엔트로피(불확실한) 예측 우선 제거
                    class_entropies = entropies[class_indices]
                    remove_indices = class_indices[np.argsort(class_entropies)[-remove_count:]]
                    
                    # 다른 클래스로 재할당
                    for idx in remove_indices:
                        probs = probabilities[idx].copy()
                        probs[class_id] = 0  # 현재 클래스 제외
                        
                        # 부족한 클래스들에 우선순위
                        for j in range(Config.N_CLASSES):
                            if current_counts[j] < target_counts[j]:
                                probs[j] *= 2
                        
                        best_class = np.argmax(probs)
                        balanced_predictions[idx] = best_class
                        current_counts[class_id] -= 1
                        current_counts[best_class] += 1
        
        return balanced_predictions
    
    def _ensemble_predictions(self, predictions_list, probabilities_list=None, method='voting'):
        """앙상블 예측 결합"""
        if len(predictions_list) == 1:
            return predictions_list[0]
        
        print(f"앙상블 예측 결합 ({method})")
        
        if method == 'voting':
            # 다수결 투표
            predictions_array = np.array(predictions_list)
            ensemble_pred = []
            
            for i in range(predictions_array.shape[1]):
                votes = predictions_array[:, i]
                unique, counts = np.unique(votes, return_counts=True)
                ensemble_pred.append(unique[np.argmax(counts)])
            
            return np.array(ensemble_pred)
        
        elif method == 'probability' and probabilities_list is not None:
            # 확률 평균
            avg_probabilities = np.mean(probabilities_list, axis=0)
            return np.argmax(avg_probabilities, axis=1)
        
        elif method == 'weighted' and probabilities_list is not None:
            # 가중 평균 (신뢰도 기반)
            weights = []
            for probs in probabilities_list:
                confidence = np.mean(np.max(probs, axis=1))
                weights.append(confidence)
            
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            
            weighted_probabilities = np.zeros_like(probabilities_list[0])
            for i, probs in enumerate(probabilities_list):
                weighted_probabilities += weights[i] * probs
            
            return np.argmax(weighted_probabilities, axis=1)
        
        else:
            # 기본적으로 첫 번째 예측 반환
            return predictions_list[0]
    
    @timer
    def create_submission_file(self, test_ids, output_path=None, predictions=None, 
                             apply_balancing=True, confidence_threshold=0.6):
        """제출 파일 생성"""
        if predictions is None:
            predictions = self.predictions
        
        if predictions is None:
            raise ValueError("예측이 수행되지 않았습니다")
        
        if output_path is None:
            output_path = Config.RESULT_FILE
        
        print("제출 파일 생성 중")
        
        # 신뢰도 기반 조정
        if self.prediction_probabilities is not None:
            predictions = self._adjust_predictions_by_confidence(
                predictions, self.prediction_probabilities, confidence_threshold
            )
        
        # 분포 균형 조정
        if apply_balancing and self.prediction_probabilities is not None:
            predictions = self._balance_prediction_distribution(
                predictions, self.prediction_probabilities
            )
        
        # 기본 검증
        if len(test_ids) != len(predictions):
            raise ValueError(f"ID 개수({len(test_ids)})와 예측 개수({len(predictions)})가 일치하지 않습니다")
        
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
            print("경고: 중복된 ID가 발견되었습니다")
        
        # 예측값 범위 확인
        invalid_predictions = (submission_df[Config.TARGET_COLUMN] < 0) | (submission_df[Config.TARGET_COLUMN] >= Config.N_CLASSES)
        if invalid_predictions.any():
            print("경고: 예측값이 유효한 범위를 벗어났습니다")
            invalid_count = invalid_predictions.sum()
            print(f"유효하지 않은 예측값 개수: {invalid_count}")
            
            # 가장 빈번한 클래스로 수정
            most_frequent_class = submission_df[Config.TARGET_COLUMN].mode()[0]
            submission_df.loc[invalid_predictions, Config.TARGET_COLUMN] = most_frequent_class
            print(f"유효하지 않은 예측값을 {most_frequent_class}으로 수정")
        
        # 파일 저장
        try:
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
                print(f"경고: 저장된 파일 크기가 다릅니다")
        except Exception as e:
            print(f"파일 검증 실패: {e}")
        
        # 예측 분포 분석
        self.analyze_prediction_distribution(predictions)
        
        return submission_df
    
    def analyze_prediction_distribution(self, predictions=None):
        """예측 분포 분석"""
        if predictions is None:
            predictions = self.predictions
        
        if predictions is None:
            print("예측이 수행되지 않았습니다")
            return None
        
        print("예측 분포 분석")
        
        unique, counts = np.unique(predictions, return_counts=True)
        total_predictions = len(predictions)
        
        print("클래스별 예측 개수:")
        for class_id in range(min(10, Config.N_CLASSES)):
            count = counts[unique == class_id][0] if class_id in unique else 0
            percentage = (count / total_predictions) * 100
            print(f"클래스 {class_id:2d}: {count:4d}개 ({percentage:5.2f}%)")
        
        if Config.N_CLASSES > 10:
            print(f"... (총 {Config.N_CLASSES}개 클래스)")
        
        # 분포 통계
        expected_per_class = total_predictions / Config.N_CLASSES
        actual_counts = [counts[unique == i][0] if i in unique else 0 for i in range(Config.N_CLASSES)]
        
        print(f"\n총 예측 개수: {total_predictions}")
        print(f"클래스당 기대 개수: {expected_per_class:.1f}")
        print(f"실제 분포 표준편차: {np.std(actual_counts):.2f}")
        
        # 불균형 정도 계산
        max_count = max(actual_counts)
        min_count = min([c for c in actual_counts if c > 0]) if any(actual_counts) else 1
        imbalance_ratio = max_count / min_count
        
        print(f"불균형 비율: {imbalance_ratio:.2f}:1")
        
        # 누락된 클래스
        missing_classes = [i for i in range(Config.N_CLASSES) if i not in unique]
        if missing_classes:
            print(f"누락된 클래스: {missing_classes}")
        
        return {
            'distribution': dict(zip(unique, counts)),
            'total_predictions': total_predictions,
            'expected_per_class': expected_per_class,
            'imbalance_ratio': imbalance_ratio,
            'missing_classes': missing_classes,
            'std': np.std(actual_counts)
        }
    
    def validate_predictions(self, y_true=None):
        """예측 결과 검증"""
        if self.predictions is None:
            print("예측이 수행되지 않았습니다")
            return None
        
        print("예측 결과 검증")
        
        print(f"예측 개수: {len(self.predictions)}")
        print(f"고유 클래스 개수: {len(np.unique(self.predictions))}")
        print(f"예측값 범위: {self.predictions.min()} ~ {self.predictions.max()}")
        
        # 신뢰도 분석
        if self.confidence_scores is not None:
            print(f"평균 신뢰도: {np.mean(self.confidence_scores):.4f}")
            print(f"신뢰도 표준편차: {np.std(self.confidence_scores):.4f}")
            
            high_conf = np.sum(self.confidence_scores > 0.8)
            medium_conf = np.sum((self.confidence_scores > 0.5) & (self.confidence_scores <= 0.8))
            low_conf = np.sum(self.confidence_scores <= 0.5)
            
            print(f"높은 신뢰도 (>0.8): {high_conf}개 ({high_conf/len(self.predictions)*100:.1f}%)")
            print(f"중간 신뢰도 (0.5-0.8): {medium_conf}개 ({medium_conf/len(self.predictions)*100:.1f}%)")
            print(f"낮은 신뢰도 (≤0.5): {low_conf}개 ({low_conf/len(self.predictions)*100:.1f}%)")
        
        # 실제 레이블이 제공된 경우 성능 계산
        if y_true is not None:
            if len(y_true) != len(self.predictions):
                print("경고: 실제 레이블과 예측값의 개수가 다릅니다")
                return None
            
            macro_f1 = calculate_macro_f1(y_true, self.predictions)
            print(f"Macro F1 Score: {macro_f1:.4f}")
            
            # 클래스별 성능 분석
            try:
                report = classification_report(y_true, self.predictions, output_dict=True, zero_division=0)
                
                class_metrics = []
                for class_id in range(Config.N_CLASSES):
                    class_key = str(class_id)
                    if class_key in report:
                        class_metrics.append({
                            'class': class_id,
                            'precision': report[class_key]['precision'],
                            'recall': report[class_key]['recall'],
                            'f1_score': report[class_key]['f1-score'],
                            'support': report[class_key]['support']
                        })
                        
                        if class_id < 10:
                            print(f"클래스 {class_id:2d} - F1: {report[class_key]['f1-score']:.4f}, 지원: {report[class_key]['support']:4d}")
                
                # 성능이 낮은 클래스 식별
                low_performance_classes = [
                    m for m in class_metrics 
                    if m['f1_score'] < Config.CLASS_PERFORMANCE_THRESHOLD and m['support'] > 0
                ]
                
                if low_performance_classes:
                    print(f"\n성능이 낮은 클래스 ({len(low_performance_classes)}개):")
                    for m in low_performance_classes[:5]:
                        print(f"  클래스 {m['class']:2d}: F1={m['f1_score']:.3f}, 지원={m['support']:4d}")
                
                return {
                    'macro_f1': macro_f1,
                    'class_metrics': class_metrics,
                    'low_performance_classes': low_performance_classes,
                    'prediction_distribution': self.analyze_prediction_distribution()
                }
                
            except Exception as e:
                print(f"성능 분석 실패: {e}")
                return {'macro_f1': macro_f1}
        
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
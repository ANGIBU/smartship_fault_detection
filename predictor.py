# predictor.py

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

from config import Config
from utils import timer, load_model, calculate_macro_f1, validate_predictions, create_submission_template

class Predictor:
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
    def predict_with_ensemble(self, models_dict, X_test, weights=None, method='weighted_average'):
        """앙상블 예측"""
        print("=== 앙상블 예측 수행 중 ===")
        print(f"사용 모델 수: {len(models_dict)}")
        print(f"앙상블 방법: {method}")
        
        if weights is None:
            weights = [1.0] * len(models_dict)
        
        if len(weights) != len(models_dict):
            print("가중치 개수가 모델 개수와 맞지 않아 균등 가중치 사용")
            weights = [1.0] * len(models_dict)
        
        all_predictions = []
        all_probabilities = []
        
        for i, (name, model) in enumerate(models_dict.items()):
            print(f"{name} 예측 중...")
            
            try:
                # 확률 예측
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_test)
                    all_probabilities.append(proba * weights[i])
                
                # 클래스 예측
                pred = model.predict(X_test)
                all_predictions.append(pred)
                
            except Exception as e:
                print(f"{name} 예측 실패: {e}")
                continue
        
        if not all_predictions:
            raise ValueError("유효한 예측이 없습니다.")
        
        # 앙상블 방법에 따른 예측
        if method == 'weighted_average' and all_probabilities:
            # 가중 평균 확률 기반 예측
            ensemble_probabilities = np.average(all_probabilities, axis=0, weights=weights[:len(all_probabilities)])
            ensemble_predictions = np.argmax(ensemble_probabilities, axis=1)
            self.prediction_probabilities = ensemble_probabilities
            
        elif method == 'majority_vote':
            # 다수결 투표
            ensemble_predictions = []
            for i in range(len(X_test)):
                votes = [pred[i] for pred in all_predictions]
                ensemble_predictions.append(max(set(votes), key=votes.count))
            ensemble_predictions = np.array(ensemble_predictions)
            
        elif method == 'rank_average':
            # 랭크 평균
            rank_matrix = np.zeros((len(X_test), Config.N_CLASSES))
            
            for pred in all_predictions:
                for i, p in enumerate(pred):
                    rank_matrix[i, p] += 1
            
            ensemble_predictions = np.argmax(rank_matrix, axis=1)
            
        else:
            # 기본: 가중 평균
            if all_probabilities:
                ensemble_probabilities = np.average(all_probabilities, axis=0, weights=weights[:len(all_probabilities)])
                ensemble_predictions = np.argmax(ensemble_probabilities, axis=1)
                self.prediction_probabilities = ensemble_probabilities
            else:
                # 확률이 없으면 다수결
                ensemble_predictions = []
                for i in range(len(X_test)):
                    votes = [pred[i] for pred in all_predictions]
                    ensemble_predictions.append(max(set(votes), key=votes.count))
                ensemble_predictions = np.array(ensemble_predictions)
        
        self.predictions = ensemble_predictions
        print(f"앙상블 예측 완료: {len(self.predictions)}개 샘플")
        
        # 예측 결과 검증
        validate_predictions(self.predictions, Config.N_CLASSES)
        
        return self.predictions
    
    @timer
    def predict_with_confidence_filtering(self, X_test, confidence_threshold=0.8):
        """신뢰도 기반 예측 필터링"""
        if self.model is None:
            raise ValueError("모델이 로드되지 않았습니다.")
        
        print(f"=== 신뢰도 기반 예측 수행 중 (임계값: {confidence_threshold}) ===")
        
        # 확률 예측
        probabilities = self.model.predict_proba(X_test)
        predictions = self.model.predict(X_test)
        
        # 최대 확률로 신뢰도 계산
        max_probabilities = np.max(probabilities, axis=1)
        high_confidence_mask = max_probabilities >= confidence_threshold
        
        print(f"신뢰도 {confidence_threshold} 이상 예측: {np.sum(high_confidence_mask)}개 ({np.mean(high_confidence_mask)*100:.1f}%)")
        print(f"낮은 신뢰도 예측: {np.sum(~high_confidence_mask)}개 ({np.mean(~high_confidence_mask)*100:.1f}%)")
        
        # 신뢰도 정보 반환
        confidence_info = {
            'predictions': predictions,
            'probabilities': probabilities,
            'max_probabilities': max_probabilities,
            'high_confidence_mask': high_confidence_mask,
            'confidence_threshold': confidence_threshold,
            'high_confidence_count': np.sum(high_confidence_mask),
            'low_confidence_count': np.sum(~high_confidence_mask)
        }
        
        return confidence_info
    
    @timer
    def predict_with_tta(self, X_test, n_augmentations=5, noise_std=0.01):
        """테스트 타임 증강을 이용한 예측"""
        if self.model is None:
            raise ValueError("모델이 로드되지 않았습니다.")
        
        print(f"=== TTA 예측 수행 중 (증강 {n_augmentations}회, 노이즈 std={noise_std}) ===")
        
        all_predictions = []
        all_probabilities = []
        
        # 원본 예측
        original_pred = self.model.predict_proba(X_test)
        all_probabilities.append(original_pred)
        
        # 노이즈 추가 예측
        np.random.seed(Config.RANDOM_STATE)
        for i in range(n_augmentations):
            noise = np.random.normal(0, noise_std, X_test.shape)
            X_test_aug = X_test + noise
            
            try:
                aug_pred = self.model.predict_proba(X_test_aug)
                all_probabilities.append(aug_pred)
            except Exception as e:
                print(f"증강 {i+1} 실패: {e}")
                continue
        
        # 평균 확률
        if all_probabilities:
            ensemble_probabilities = np.mean(all_probabilities, axis=0)
            ensemble_predictions = np.argmax(ensemble_probabilities, axis=1)
            
            self.predictions = ensemble_predictions
            self.prediction_probabilities = ensemble_probabilities
            
            print(f"TTA 예측 완료: {len(all_probabilities)}개 증강 사용")
        else:
            print("TTA 실패, 원본 예측 사용")
            self.predictions = self.model.predict(X_test)
        
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
        for class_id in sorted(distribution.keys()):
            percentage = (distribution[class_id] / len(self.predictions)) * 100
            print(f"클래스 {class_id}: {distribution[class_id]}개 ({percentage:.2f}%)")
        
        # 분포 균형도 확인
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
                imbalance_scores.append(1.0)  # 완전 불균형
            else:
                imbalance = abs(count - expected_per_class) / expected_per_class
                imbalance_scores.append(imbalance)
        
        avg_imbalance = np.mean(imbalance_scores)
        print(f"평균 불균형 정도: {avg_imbalance:.3f}")
        
        if missing_classes:
            print(f"경고: 예측되지 않은 클래스: {missing_classes}")
        
        return {
            'distribution': distribution,
            'total_predictions': total_predictions,
            'expected_per_class': expected_per_class,
            'avg_imbalance': avg_imbalance,
            'missing_classes': missing_classes
        }
    
    @timer
    def create_submission_file(self, test_ids, output_path=None, predictions=None):
        """제출 파일 생성"""
        if predictions is None:
            predictions = self.predictions
        
        if predictions is None:
            raise ValueError("예측이 수행되지 않았습니다.")
        
        if output_path is None:
            output_path = Config.RESULT_FILE
        
        print("=== 제출 파일 생성 중 ===")
        
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
            if not saved_df.equals(submission_df):
                print("경고: 저장된 파일이 원본과 다릅니다.")
            else:
                print("파일 저장 검증 완료")
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
            
            # 클래스별 정확도
            class_accuracies = []
            class_f1_scores = []
            
            for class_id in range(Config.N_CLASSES):
                class_mask = y_true == class_id
                if np.sum(class_mask) > 0:
                    class_pred = self.predictions[class_mask]
                    accuracy = np.mean(class_pred == class_id)
                    
                    # 클래스별 F1 스코어
                    true_pos = np.sum((y_true == class_id) & (self.predictions == class_id))
                    false_pos = np.sum((y_true != class_id) & (self.predictions == class_id))
                    false_neg = np.sum((y_true == class_id) & (self.predictions != class_id))
                    
                    if true_pos + false_pos > 0:
                        precision = true_pos / (true_pos + false_pos)
                    else:
                        precision = 0
                    
                    if true_pos + false_neg > 0:
                        recall = true_pos / (true_pos + false_neg)
                    else:
                        recall = 0
                    
                    if precision + recall > 0:
                        f1 = 2 * (precision * recall) / (precision + recall)
                    else:
                        f1 = 0
                    
                    class_accuracies.append(accuracy)
                    class_f1_scores.append(f1)
                    
                    print(f"클래스 {class_id} - 정확도: {accuracy:.4f}, F1: {f1:.4f}")
            
            avg_class_accuracy = np.mean(class_accuracies) if class_accuracies else 0
            avg_class_f1 = np.mean(class_f1_scores) if class_f1_scores else 0
            
            print(f"평균 클래스별 정확도: {avg_class_accuracy:.4f}")
            print(f"평균 클래스별 F1: {avg_class_f1:.4f}")
            
            return {
                'macro_f1': macro_f1,
                'avg_class_accuracy': avg_class_accuracy,
                'avg_class_f1': avg_class_f1,
                'class_accuracies': class_accuracies,
                'class_f1_scores': class_f1_scores
            }
        
        return None
    
    @timer
    def calibrate_predictions(self, X_cal, y_cal, method='isotonic'):
        """예측 확률 보정"""
        if self.model is None:
            raise ValueError("모델이 로드되지 않았습니다.")
        
        print(f"=== 예측 확률 보정 ({method}) ===")
        
        if method == 'isotonic':
            from sklearn.isotonic import IsotonicRegression
            calibrator = IsotonicRegression(out_of_bounds='clip')
        elif method == 'platt':
            from sklearn.linear_model import LogisticRegression
            calibrator = LogisticRegression(random_state=Config.RANDOM_STATE)
        else:
            raise ValueError("지원하지 않는 보정 방법입니다.")
        
        # 보정 데이터로 확률 계산
        cal_proba = self.model.predict_proba(X_cal)
        
        # 클래스별 보정기 훈련
        calibrated_classifiers = {}
        for class_id in range(Config.N_CLASSES):
            # 이진 분류 문제로 변환
            binary_labels = (y_cal == class_id).astype(int)
            class_proba = cal_proba[:, class_id]
            
            if method == 'isotonic':
                calibrator_class = IsotonicRegression(out_of_bounds='clip')
                calibrator_class.fit(class_proba, binary_labels)
            else:
                calibrator_class = LogisticRegression(random_state=Config.RANDOM_STATE)
                calibrator_class.fit(class_proba.reshape(-1, 1), binary_labels)
            
            calibrated_classifiers[class_id] = calibrator_class
        
        self.calibrated_classifiers = calibrated_classifiers
        print("확률 보정 훈련 완료")
        
        return calibrated_classifiers
    
    def predict_calibrated(self, X_test):
        """보정된 확률로 예측"""
        if not hasattr(self, 'calibrated_classifiers'):
            raise ValueError("확률 보정이 수행되지 않았습니다.")
        
        print("=== 보정된 확률 예측 ===")
        
        # 원본 확률
        original_proba = self.model.predict_proba(X_test)
        calibrated_proba = np.zeros_like(original_proba)
        
        # 클래스별 보정 적용
        for class_id in range(Config.N_CLASSES):
            calibrator = self.calibrated_classifiers[class_id]
            class_proba = original_proba[:, class_id]
            
            if hasattr(calibrator, 'predict_proba'):
                # Platt scaling
                calibrated_class_proba = calibrator.predict_proba(class_proba.reshape(-1, 1))[:, 1]
            else:
                # Isotonic regression
                calibrated_class_proba = calibrator.predict(class_proba)
            
            calibrated_proba[:, class_id] = calibrated_class_proba
        
        # 확률 정규화
        calibrated_proba = calibrated_proba / calibrated_proba.sum(axis=1, keepdims=True)
        
        # 예측
        calibrated_predictions = np.argmax(calibrated_proba, axis=1)
        
        self.prediction_probabilities = calibrated_proba
        self.predictions = calibrated_predictions
        
        print("보정된 확률 예측 완료")
        return calibrated_predictions
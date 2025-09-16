# predictor.py

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

from config import Config
from utils import timer, load_model, calculate_macro_f1

class Predictor:
    def __init__(self, model=None):
        self.model = model
        self.predictions = None
        self.prediction_probabilities = None
        
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
    def predict(self, X_test):
        """예측 수행"""
        if self.model is None:
            raise ValueError("모델이 로드되지 않았습니다.")
        
        print("=== 예측 수행 중 ===")
        
        # 예측 확률
        if hasattr(self.model, 'predict_proba'):
            self.prediction_probabilities = self.model.predict_proba(X_test)
            print(f"예측 확률 형태: {self.prediction_probabilities.shape}")
        
        # 예측 클래스
        self.predictions = self.model.predict(X_test)
        print(f"예측 결과 형태: {self.predictions.shape}")
        
        return self.predictions
    
    @timer
    def predict_with_ensemble(self, models_dict, X_test, weights=None):
        """앙상블 예측"""
        print("=== 앙상블 예측 수행 중 ===")
        
        if weights is None:
            weights = [1.0] * len(models_dict)
        
        all_predictions = []
        all_probabilities = []
        
        for i, (name, model) in enumerate(models_dict.items()):
            print(f"{name} 예측 중...")
            
            # 확률 예측
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_test)
                all_probabilities.append(proba * weights[i])
            
            # 클래스 예측
            pred = model.predict(X_test)
            all_predictions.append(pred)
        
        # 가중 평균 확률 기반 예측
        if all_probabilities:
            ensemble_probabilities = np.mean(all_probabilities, axis=0)
            ensemble_predictions = np.argmax(ensemble_probabilities, axis=1)
            self.prediction_probabilities = ensemble_probabilities
        else:
            # 다수결 투표
            ensemble_predictions = []
            for i in range(len(X_test)):
                votes = [pred[i] for pred in all_predictions]
                ensemble_predictions.append(max(set(votes), key=votes.count))
            ensemble_predictions = np.array(ensemble_predictions)
        
        self.predictions = ensemble_predictions
        print(f"앙상블 예측 완료: {len(self.predictions)}개 샘플")
        
        return self.predictions
    
    @timer
    def predict_with_confidence(self, X_test, confidence_threshold=0.8):
        """신뢰도 기반 예측"""
        if self.model is None:
            raise ValueError("모델이 로드되지 않았습니다.")
        
        print("=== 신뢰도 기반 예측 수행 중 ===")
        
        # 확률 예측
        probabilities = self.model.predict_proba(X_test)
        predictions = self.model.predict(X_test)
        
        # 최대 확률로 신뢰도 계산
        max_probabilities = np.max(probabilities, axis=1)
        high_confidence_mask = max_probabilities >= confidence_threshold
        
        print(f"신뢰도 {confidence_threshold} 이상 예측: {np.sum(high_confidence_mask)}개")
        print(f"낮은 신뢰도 예측: {np.sum(~high_confidence_mask)}개")
        
        # 신뢰도 정보 추가
        confidence_info = {
            'predictions': predictions,
            'probabilities': probabilities,
            'max_probabilities': max_probabilities,
            'high_confidence_mask': high_confidence_mask,
            'confidence_threshold': confidence_threshold
        }
        
        return confidence_info
    
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
            print(f"클래스 {class_id}: {distribution[class_id]}개")
        
        # 분포 균형도 확인
        total_predictions = len(self.predictions)
        expected_per_class = total_predictions / Config.N_CLASSES
        
        print(f"\n총 예측 개수: {total_predictions}")
        print(f"클래스당 기대 개수: {expected_per_class:.1f}")
        
        # 불균형 정도 계산
        imbalance_scores = []
        for class_id in range(Config.N_CLASSES):
            count = distribution.get(class_id, 0)
            imbalance = abs(count - expected_per_class) / expected_per_class
            imbalance_scores.append(imbalance)
        
        avg_imbalance = np.mean(imbalance_scores)
        print(f"평균 불균형 정도: {avg_imbalance:.3f}")
        
        return distribution
    
    @timer
    def create_submission_file(self, test_ids, output_path=None):
        """제출 파일 생성"""
        if self.predictions is None:
            raise ValueError("예측이 수행되지 않았습니다.")
        
        if output_path is None:
            output_path = Config.RESULT_FILE
        
        print("=== 제출 파일 생성 중 ===")
        
        submission_df = pd.DataFrame({
            Config.ID_COLUMN: test_ids,
            Config.TARGET_COLUMN: self.predictions
        })
        
        # 데이터 검증
        print(f"제출 파일 형태: {submission_df.shape}")
        print(f"ID 개수: {len(submission_df[Config.ID_COLUMN].unique())}")
        print(f"예측값 범위: {submission_df[Config.TARGET_COLUMN].min()} ~ {submission_df[Config.TARGET_COLUMN].max()}")
        
        # 중복 ID 확인
        if submission_df[Config.ID_COLUMN].duplicated().any():
            print("경고: 중복된 ID가 발견되었습니다.")
        
        # 예측값 범위 확인
        if (submission_df[Config.TARGET_COLUMN] < 0).any() or (submission_df[Config.TARGET_COLUMN] >= Config.N_CLASSES).any():
            print("경고: 예측값이 유효한 범위를 벗어났습니다.")
        
        # 파일 저장
        submission_df.to_csv(output_path, index=False)
        print(f"제출 파일 저장 완료: {output_path}")
        
        # 저장된 파일 검증
        saved_df = pd.read_csv(output_path)
        if not saved_df.equals(submission_df):
            print("경고: 저장된 파일이 원본과 다릅니다.")
        else:
            print("파일 저장 검증 완료")
        
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
            for class_id in range(Config.N_CLASSES):
                class_mask = y_true == class_id
                if np.sum(class_mask) > 0:
                    class_pred = self.predictions[class_mask]
                    accuracy = np.mean(class_pred == class_id)
                    class_accuracies.append(accuracy)
                    print(f"클래스 {class_id} 정확도: {accuracy:.4f}")
            
            avg_class_accuracy = np.mean(class_accuracies)
            print(f"평균 클래스별 정확도: {avg_class_accuracy:.4f}")
            
            return macro_f1
        
        return None
    
    @timer
    def predict_with_tta(self, X_test, n_augmentations=5, noise_std=0.01):
        """테스트 타임 증강을 이용한 예측"""
        if self.model is None:
            raise ValueError("모델이 로드되지 않았습니다.")
        
        print(f"=== TTA 예측 수행 중 (증강 {n_augmentations}회) ===")
        
        all_predictions = []
        
        # 원본 예측
        original_pred = self.model.predict_proba(X_test)
        all_predictions.append(original_pred)
        
        # 노이즈 추가 예측
        for i in range(n_augmentations):
            noise = np.random.normal(0, noise_std, X_test.shape)
            X_test_aug = X_test + noise
            
            aug_pred = self.model.predict_proba(X_test_aug)
            all_predictions.append(aug_pred)
        
        # 평균 확률
        ensemble_probabilities = np.mean(all_predictions, axis=0)
        ensemble_predictions = np.argmax(ensemble_probabilities, axis=1)
        
        self.predictions = ensemble_predictions
        self.prediction_probabilities = ensemble_probabilities
        
        print(f"TTA 예측 완료: {len(self.predictions)}개 샘플")
        
        return self.predictions
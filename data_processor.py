# data_processor.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler, QuantileTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif, chi2
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from scipy.stats import skew, kurtosis
import gc
import warnings
warnings.filterwarnings('ignore')

from config import Config
from utils import timer, check_data_quality, save_joblib

class DataProcessor:
    def __init__(self):
        self.scaler = None
        self.feature_selector = None
        self.imputer = None
        self.feature_columns = Config.FEATURE_COLUMNS
        self.selected_features = None
        self.feature_importance = {}
        
    @timer
    def load_data(self):
        """데이터 로드"""
        print("데이터 로드 시작")
        
        try:
            train_df = pd.read_csv(Config.TRAIN_FILE)
            test_df = pd.read_csv(Config.TEST_FILE)
            
            print(f"Train 데이터 형태: {train_df.shape}")
            print(f"Test 데이터 형태: {test_df.shape}")
            
            # 데이터 타입 최적화
            for col in self.feature_columns:
                if col in train_df.columns:
                    train_df[col] = train_df[col].astype('float32')
                if col in test_df.columns:
                    test_df[col] = test_df[col].astype('float32')
            
            if Config.TARGET_COLUMN in train_df.columns:
                train_df[Config.TARGET_COLUMN] = train_df[Config.TARGET_COLUMN].astype('int16')
            
            # 데이터 품질 검사
            train_quality = check_data_quality(train_df, self.feature_columns)
            test_quality = check_data_quality(test_df, self.feature_columns)
            
            if not (train_quality and test_quality):
                print("데이터 품질 문제 처리 중")
                train_df, test_df = self._handle_data_issues(train_df, test_df)
            
            return train_df, test_df
            
        except Exception as e:
            print(f"데이터 로드 실패: {e}")
            raise
    
    def _handle_data_issues(self, train_df, test_df):
        """데이터 이슈 처리"""
        for col in self.feature_columns:
            if col in train_df.columns and col in test_df.columns:
                # 무한값 처리
                train_df[col] = train_df[col].replace([np.inf, -np.inf], np.nan)
                test_df[col] = test_df[col].replace([np.inf, -np.inf], np.nan)
                
                # 결측치 처리
                if train_df[col].isnull().sum() > 0 or test_df[col].isnull().sum() > 0:
                    fill_val = train_df[col].median()
                    if pd.isna(fill_val):
                        fill_val = 0.0
                    
                    train_df[col].fillna(fill_val, inplace=True)
                    test_df[col].fillna(fill_val, inplace=True)
        
        return train_df, test_df
    
    @timer
    def create_statistical_features(self, X_train, X_test):
        """통계적 피처 생성"""
        print("통계적 피처 생성 중")
        
        for df in [X_train, X_test]:
            sensor_data = df[self.feature_columns].values
            
            # 기본 통계량
            df['sensor_mean'] = np.mean(sensor_data, axis=1).astype('float32')
            df['sensor_std'] = np.std(sensor_data, axis=1).astype('float32')
            df['sensor_median'] = np.median(sensor_data, axis=1).astype('float32')
            df['sensor_min'] = np.min(sensor_data, axis=1).astype('float32')
            df['sensor_max'] = np.max(sensor_data, axis=1).astype('float32')
            df['sensor_range'] = (df['sensor_max'] - df['sensor_min']).astype('float32')
            
            # 분포 특성
            df['sensor_skew'] = skew(sensor_data, axis=1, nan_policy='omit').astype('float32')
            df['sensor_kurtosis'] = kurtosis(sensor_data, axis=1, nan_policy='omit').astype('float32')
            
            # 분위수
            df['sensor_q25'] = np.percentile(sensor_data, 25, axis=1).astype('float32')
            df['sensor_q75'] = np.percentile(sensor_data, 75, axis=1).astype('float32')
            df['sensor_iqr'] = (df['sensor_q75'] - df['sensor_q25']).astype('float32')
            
            # 변동성 지표
            df['sensor_cv'] = (df['sensor_std'] / (np.abs(df['sensor_mean']) + 1e-8)).astype('float32')
            
            # 이상치 개수
            q75 = df['sensor_q75'].values.reshape(-1, 1)
            q25 = df['sensor_q25'].values.reshape(-1, 1)
            iqr = (q75 - q25)
            
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            
            outliers = ((sensor_data < lower_bound) | (sensor_data > upper_bound)).sum(axis=1)
            df['sensor_outlier_count'] = outliers.astype('float32')
            df['sensor_outlier_ratio'] = (outliers / len(self.feature_columns)).astype('float32')
        
        return X_train, X_test
    
    @timer
    def create_correlation_features(self, X_train, X_test):
        """상관관계 기반 피처 생성"""
        print("상관관계 피처 생성 중")
        
        # 훈련 데이터에서 상관관계 계산
        train_corr = X_train[self.feature_columns].corr()
        
        # 높은 상관관계를 가진 센서 쌍 식별
        high_corr_pairs = []
        for i in range(len(self.feature_columns)):
            for j in range(i+1, len(self.feature_columns)):
                corr_val = train_corr.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr_pairs.append((self.feature_columns[i], self.feature_columns[j], corr_val))
        
        print(f"높은 상관관계 센서 쌍: {len(high_corr_pairs)}개")
        
        # 상관관계 기반 피처 생성
        for df in [X_train, X_test]:
            for sensor1, sensor2, corr_val in high_corr_pairs[:10]:  # 상위 10개만
                if sensor1 in df.columns and sensor2 in df.columns:
                    # 비율
                    df[f'{sensor1}_{sensor2}_ratio'] = (df[sensor1] / (df[sensor2] + 1e-8)).astype('float32')
                    # 차이
                    df[f'{sensor1}_{sensor2}_diff'] = (df[sensor1] - df[sensor2]).astype('float32')
        
        return X_train, X_test
    
    @timer
    def handle_class_imbalance(self, X_train, y_train):
        """클래스 불균형 처리"""
        print("클래스 불균형 처리 시작")
        
        # 클래스 분포 확인
        class_counts = y_train.value_counts().sort_index()
        print("클래스별 샘플 수:")
        for class_id, count in class_counts.items():
            if class_id < 10:
                print(f"  클래스 {class_id}: {count}개")
        
        # 불균형 정도 계산
        max_count = class_counts.max()
        min_count = class_counts.min()
        imbalance_ratio = max_count / min_count
        
        print(f"불균형 비율: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 3.0:
            print("SMOTE 적용")
            
            try:
                # 적절한 k_neighbors 계산
                min_class_samples = class_counts.min()
                k_neighbors = min(5, min_class_samples - 1)
                
                if k_neighbors < 1:
                    print("SMOTE 적용 불가, 원본 데이터 사용")
                    return X_train, y_train
                
                smote = SMOTE(
                    sampling_strategy='auto',
                    random_state=Config.RANDOM_STATE,
                    k_neighbors=k_neighbors
                )
                
                X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
                
                print(f"리샘플링 전: {X_train.shape}")
                print(f"리샘플링 후: {X_resampled.shape}")
                
                return X_resampled, y_resampled
                
            except Exception as e:
                print(f"SMOTE 적용 실패: {e}")
                return X_train, y_train
        else:
            print("클래스 균형이 적절하여 리샘플링 생략")
            return X_train, y_train
    
    @timer
    def scale_features(self, X_train, X_test, method='robust'):
        """피처 스케일링"""
        print(f"피처 스케일링 시작 ({method})")
        
        if method == 'robust':
            self.scaler = RobustScaler()
        elif method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'quantile':
            self.scaler = QuantileTransformer(
                output_distribution='normal',
                random_state=Config.RANDOM_STATE,
                n_quantiles=min(1000, X_train.shape[0])
            )
        else:
            self.scaler = RobustScaler()
        
        # 수치형 컬럼만 스케일링
        numeric_columns = X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        # 스케일링 적용
        X_train_scaled[numeric_columns] = self.scaler.fit_transform(X_train[numeric_columns])
        X_test_scaled[numeric_columns] = self.scaler.transform(X_test[numeric_columns])
        
        # 데이터 타입 유지
        for col in numeric_columns:
            X_train_scaled[col] = X_train_scaled[col].astype('float32')
            X_test_scaled[col] = X_test_scaled[col].astype('float32')
        
        save_joblib(self.scaler, Config.SCALER_FILE)
        
        return X_train_scaled, X_test_scaled
    
    @timer
    def select_features(self, X_train, X_test, y_train, n_features=None):
        """피처 선택"""
        if n_features is None:
            n_features = Config.TARGET_FEATURES
        
        n_features = min(n_features, X_train.shape[1])
        
        print(f"피처 선택 시작 (목표: {n_features}개)")
        print(f"원본 피처 개수: {X_train.shape[1]}")
        
        # 분산이 0인 피처 제거
        constant_features = []
        for col in X_train.columns:
            if X_train[col].nunique() <= 1:
                constant_features.append(col)
        
        if constant_features:
            print(f"상수 피처 제거: {len(constant_features)}개")
            X_train = X_train.drop(columns=constant_features)
            X_test = X_test.drop(columns=constant_features)
        
        # 다중 피처 선택 방법 적용
        feature_scores = {}
        
        # Mutual Information
        try:
            mi_selector = SelectKBest(mutual_info_classif, k=n_features)
            mi_selector.fit(X_train, y_train)
            mi_scores = mi_selector.scores_
            
            for i, col in enumerate(X_train.columns):
                if col not in feature_scores:
                    feature_scores[col] = 0
                feature_scores[col] += mi_scores[i]
        except Exception as e:
            print(f"Mutual Information 계산 실패: {e}")
        
        # F-test
        try:
            f_selector = SelectKBest(f_classif, k=n_features)
            f_selector.fit(X_train, y_train)
            f_scores = f_selector.scores_
            
            for i, col in enumerate(X_train.columns):
                if col not in feature_scores:
                    feature_scores[col] = 0
                feature_scores[col] += f_scores[i] / np.max(f_scores)  # 정규화
        except Exception as e:
            print(f"F-test 계산 실패: {e}")
        
        # 점수 기반 피처 선택
        if feature_scores:
            sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
            self.selected_features = [feat for feat, score in sorted_features[:n_features]]
            
            # 피처 중요도 저장
            self.feature_importance = dict(sorted_features[:n_features])
        else:
            # 기본적으로 원본 센서 피처 사용
            available_sensors = [col for col in self.feature_columns if col in X_train.columns]
            self.selected_features = available_sensors[:n_features]
            print("피처 선택 실패, 원본 센서 피처 사용")
        
        X_train_selected = X_train[self.selected_features].copy()
        X_test_selected = X_test[self.selected_features].copy()
        
        print(f"선택된 피처 개수: {len(self.selected_features)}")
        
        # 선택된 피처 유형 분석
        original_count = sum(1 for f in self.selected_features if f in self.feature_columns)
        statistical_count = sum(1 for f in self.selected_features if 'sensor_' in f)
        correlation_count = len(self.selected_features) - original_count - statistical_count
        
        print(f"피처 유형별 선택:")
        print(f"  원본 센서: {original_count}개")
        print(f"  통계적: {statistical_count}개")
        print(f"  상관관계: {correlation_count}개")
        
        return X_train_selected, X_test_selected
    
    @timer
    def get_processed_data(self, use_resampling=True, scaling_method='robust'):
        """전체 전처리 파이프라인 실행"""
        print("전체 데이터 전처리 파이프라인 시작")
        
        try:
            # 1. 데이터 로드
            train_df, test_df = self.load_data()
            
            # ID 컬럼 저장
            train_ids = train_df[Config.ID_COLUMN].copy()
            test_ids = test_df[Config.ID_COLUMN].copy()
            
            # 피처와 타겟 분리
            X_train = train_df[self.feature_columns].copy()
            X_test = test_df[self.feature_columns].copy()
            y_train = train_df[Config.TARGET_COLUMN].copy()
            
            del train_df, test_df
            gc.collect()
            
            # 2. 통계적 피처 생성
            X_train, X_test = self.create_statistical_features(X_train, X_test)
            
            # 3. 상관관계 피처 생성
            X_train, X_test = self.create_correlation_features(X_train, X_test)
            
            # 4. 스케일링
            X_train, X_test = self.scale_features(X_train, X_test, scaling_method)
            
            # 5. 피처 선택
            X_train, X_test = self.select_features(X_train, X_test, y_train)
            
            # 6. 클래스 불균형 처리
            if use_resampling:
                X_train, y_train = self.handle_class_imbalance(X_train, y_train)
                # 리샘플링 후 ID 조정
                if len(X_train) != len(train_ids):
                    train_ids = pd.Series(range(len(X_train)), name=Config.ID_COLUMN)
            
            # 7. 최종 검증
            print("최종 데이터 검증")
            
            train_nan_count = X_train.isna().sum().sum()
            test_nan_count = X_test.isna().sum().sum()
            
            print(f"최종 훈련 데이터 NaN: {train_nan_count}")
            print(f"최종 테스트 데이터 NaN: {test_nan_count}")
            
            if train_nan_count > 0 or test_nan_count > 0:
                X_train.fillna(0, inplace=True)
                X_test.fillna(0, inplace=True)
                print("잔여 NaN을 0으로 대체")
            
            print(f"최종 피처 개수: {X_train.shape[1]}")
            print(f"최종 데이터 형태 - 훈련: {X_train.shape}, 테스트: {X_test.shape}")
            
            gc.collect()
            
            return X_train, X_test, y_train, train_ids, test_ids
            
        except Exception as e:
            print(f"데이터 전처리 중 오류 발생: {e}")
            gc.collect()
            raise
    
    def get_feature_importance(self):
        """피처 중요도 반환"""
        return self.feature_importance
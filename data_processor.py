# data_processor.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler, QuantileTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
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
        self.train_stats = {}
        self.sensor_groups = Config.SENSOR_GROUPS
        self.generated_feature_names = []
        
    @timer
    def load_and_preprocess_data(self):
        """데이터 로드 및 전처리"""
        print("데이터 로드 시작")
        
        # 메모리 효율적 로드
        dtypes = {col: 'float32' for col in self.feature_columns}
        
        train_df = pd.read_csv(Config.TRAIN_FILE, dtype=dtypes)
        test_df = pd.read_csv(Config.TEST_FILE, dtype=dtypes)
        
        print(f"Train 데이터 형태: {train_df.shape}")
        print(f"Test 데이터 형태: {test_df.shape}")
        
        # 타겟 컬럼 처리
        if Config.TARGET_COLUMN in train_df.columns:
            train_df[Config.TARGET_COLUMN] = train_df[Config.TARGET_COLUMN].astype('int32')
        
        # 데이터 품질 검사
        train_quality = check_data_quality(train_df, self.feature_columns)
        test_quality = check_data_quality(test_df, self.feature_columns)
        
        if not (train_quality and test_quality):
            print("데이터 품질 문제 처리 중")
            train_df, test_df = self._handle_data_issues(train_df, test_df)
        
        return train_df, test_df
    
    def _handle_data_issues(self, train_df, test_df):
        """데이터 이슈 처리"""
        for col in self.feature_columns:
            # 무한값을 NaN으로 변환
            train_df[col] = train_df[col].replace([np.inf, -np.inf], np.nan)
            test_df[col] = test_df[col].replace([np.inf, -np.inf], np.nan)
            
            # 결측치 처리
            if train_df[col].isnull().sum() > 0:
                fill_val = train_df[col].median()
                if pd.isna(fill_val):
                    fill_val = 0.0
                    
                train_df[col].fillna(fill_val, inplace=True)
                test_df[col].fillna(fill_val, inplace=True)
        
        return train_df, test_df
    
    def _create_basic_statistics(self, X_train, X_test):
        """기본 통계 피처 생성"""
        print("기본 통계 피처 생성 중")
        
        for df in [X_train, X_test]:
            arr = df[self.feature_columns].values
            
            # 기본 통계량
            df['sensor_mean'] = np.mean(arr, axis=1).astype('float32')
            df['sensor_std'] = np.std(arr, axis=1).astype('float32')
            df['sensor_median'] = np.median(arr, axis=1).astype('float32')
            df['sensor_min'] = np.min(arr, axis=1).astype('float32')
            df['sensor_max'] = np.max(arr, axis=1).astype('float32')
            df['sensor_range'] = (df['sensor_max'] - df['sensor_min']).astype('float32')
            
            # 분위수
            df['sensor_q25'] = np.percentile(arr, 25, axis=1).astype('float32')
            df['sensor_q75'] = np.percentile(arr, 75, axis=1).astype('float32')
            df['sensor_iqr'] = (df['sensor_q75'] - df['sensor_q25']).astype('float32')
            
            # 형태 통계량
            df['sensor_skew'] = skew(arr, axis=1, nan_policy='omit').astype('float32')
            df['sensor_kurtosis'] = kurtosis(arr, axis=1, nan_policy='omit').astype('float32')
        
        return X_train, X_test
    
    def _create_group_statistics(self, X_train, X_test):
        """센서 그룹별 통계 피처 생성"""
        print("센서 그룹 통계 피처 생성 중")
        
        for group_name, sensors in self.sensor_groups.items():
            valid_sensors = [s for s in sensors if s in self.feature_columns]
            if len(valid_sensors) >= 2:
                for df in [X_train, X_test]:
                    group_data = df[valid_sensors].values
                    
                    df[f'{group_name}_mean'] = np.mean(group_data, axis=1).astype('float32')
                    df[f'{group_name}_std'] = np.std(group_data, axis=1).astype('float32')
                    df[f'{group_name}_max'] = np.max(group_data, axis=1).astype('float32')
                    df[f'{group_name}_min'] = np.min(group_data, axis=1).astype('float32')
        
        return X_train, X_test
    
    @timer
    def feature_engineering(self, train_df, test_df):
        """피처 엔지니어링"""
        print("피처 엔지니어링 시작")
        
        # 메모리 효율적 복사
        X_train = train_df[self.feature_columns].copy()
        X_test = test_df[self.feature_columns].copy()
        y_train = train_df[Config.TARGET_COLUMN].copy()
        
        # 기본 통계 피처 생성
        if Config.STATISTICAL_FEATURES:
            X_train, X_test = self._create_basic_statistics(X_train, X_test)
            X_train, X_test = self._create_group_statistics(X_train, X_test)
        
        # 최종 데이터 정리
        X_train, X_test = self._final_cleanup(X_train, X_test)
        
        return X_train, X_test, y_train
    
    def _final_cleanup(self, X_train, X_test):
        """최종 데이터 정리"""
        print("데이터 정리 중")
        
        # 무한값과 NaN 처리
        for df in [X_train, X_test]:
            # 무한값을 NaN으로 변환
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            # NaN을 중앙값으로 대체
            for col in df.columns:
                if df[col].isna().any():
                    if df[col].dtype in ['float32', 'float64']:
                        fill_value = df[col].median()
                        if pd.isna(fill_value):
                            fill_value = 0.0
                        df[col].fillna(fill_value, inplace=True)
            
            # 데이터 타입 최적화
            for col in df.select_dtypes(include=['float64']).columns:
                df[col] = df[col].astype('float32')
        
        print(f"생성된 피처 수: {X_train.shape[1]}")
        
        # 생성된 피처명 저장
        self.generated_feature_names = X_train.columns.tolist()
        
        # 메모리 정리
        gc.collect()
        
        return X_train, X_test
    
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
        
        # 스케일링 수행
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # DataFrame 변환
        X_train_scaled = pd.DataFrame(
            X_train_scaled, 
            columns=X_train.columns, 
            index=X_train.index,
            dtype='float32'
        )
        X_test_scaled = pd.DataFrame(
            X_test_scaled, 
            columns=X_test.columns, 
            index=X_test.index,
            dtype='float32'
        )
        
        save_joblib(self.scaler, Config.SCALER_FILE)
        
        return X_train_scaled, X_test_scaled
    
    @timer
    def select_features(self, X_train, X_test, y_train, method='mutual_info', k=None):
        """피처 선택"""
        if k is None:
            k = Config.FEATURE_SELECTION_K
        
        k = min(k, X_train.shape[1])
        
        print(f"피처 선택 시작 ({method}, k={k})")
        print(f"원본 피처 개수: {X_train.shape[1]}")
        
        if method == 'mutual_info':
            selector = SelectKBest(mutual_info_classif, k=k)
        elif method == 'f_classif':
            selector = SelectKBest(f_classif, k=k)
        elif method == 'random_forest':
            rf = RandomForestClassifier(
                n_estimators=100, 
                random_state=Config.RANDOM_STATE,
                n_jobs=2,
                max_depth=5
            )
            rf.fit(X_train, y_train)
            importances = rf.feature_importances_
            indices = np.argsort(importances)[::-1][:k]
            
            X_train_selected = X_train.iloc[:, indices]
            X_test_selected = X_test.iloc[:, indices]
            self.selected_features = X_train.columns[indices].tolist()
            
            print(f"선택된 피처 개수: {len(self.selected_features)}")
            return X_train_selected, X_test_selected
        else:
            selector = SelectKBest(mutual_info_classif, k=k)
        
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        selected_mask = selector.get_support()
        self.selected_features = X_train.columns[selected_mask].tolist()
        
        X_train_selected = pd.DataFrame(
            X_train_selected, 
            columns=self.selected_features, 
            index=X_train.index,
            dtype='float32'
        )
        X_test_selected = pd.DataFrame(
            X_test_selected, 
            columns=self.selected_features, 
            index=X_test.index,
            dtype='float32'
        )
        
        self.feature_selector = selector
        save_joblib(self.feature_selector, Config.FEATURE_SELECTOR_FILE)
        
        print(f"선택된 피처 개수: {len(self.selected_features)}")
        
        # 메모리 정리
        gc.collect()
        
        return X_train_selected, X_test_selected
    
    def get_processed_data(self, use_feature_selection=True, scaling_method='robust'):
        """전체 전처리 파이프라인 실행"""
        print("전체 데이터 전처리 파이프라인 시작")
        
        try:
            # 1. 데이터 로드
            train_df, test_df = self.load_and_preprocess_data()
            
            # 2. 피처 엔지니어링
            X_train, X_test, y_train = self.feature_engineering(train_df, test_df)
            
            # 메모리 정리
            del train_df, test_df
            gc.collect()
            
            # 3. 스케일링
            X_train, X_test = self.scale_features(X_train, X_test, scaling_method)
            
            # 4. 피처 선택
            if use_feature_selection:
                available_features = min(Config.FEATURE_SELECTION_K, X_train.shape[1])
                X_train, X_test = self.select_features(
                    X_train, X_test, y_train, 
                    method='mutual_info', 
                    k=available_features
                )
            
            # 5. 최종 검증 및 정리
            print("최종 데이터 검증")
            
            # NaN 체크
            train_nan_count = X_train.isna().sum().sum()
            test_nan_count = X_test.isna().sum().sum()
            
            print(f"최종 훈련 데이터 NaN: {train_nan_count}")
            print(f"최종 테스트 데이터 NaN: {test_nan_count}")
            
            # 남은 NaN 처리
            if train_nan_count > 0 or test_nan_count > 0:
                X_train.fillna(0, inplace=True)
                X_test.fillna(0, inplace=True)
                print("잔여 NaN을 0으로 대체")
            
            print(f"최종 피처 개수: {X_train.shape[1]}")
            print(f"최종 데이터 형태 - 훈련: {X_train.shape}, 테스트: {X_test.shape}")
            
            # 메모리 최종 정리
            gc.collect()
            
            # ID 컬럼 반환을 위한 원본 로드
            train_ids = pd.read_csv(Config.TRAIN_FILE, usecols=[Config.ID_COLUMN])[Config.ID_COLUMN]
            test_ids = pd.read_csv(Config.TEST_FILE, usecols=[Config.ID_COLUMN])[Config.ID_COLUMN]
            
            return X_train, X_test, y_train, train_ids, test_ids
            
        except Exception as e:
            print(f"데이터 전처리 중 오류 발생: {e}")
            # 메모리 정리
            gc.collect()
            raise
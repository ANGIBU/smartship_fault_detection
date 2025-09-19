# data_processor.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler, QuantileTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
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
        self.pca = None
        self.feature_columns = Config.FEATURE_COLUMNS
        self.selected_features = None
        self.train_stats = {}
        
    @timer
    def load_and_preprocess_data(self):
        """데이터 로드 및 전처리"""
        print("=== 데이터 로드 및 전처리 시작 ===")
        
        # 메모리 효율적 로드
        train_df = pd.read_csv(Config.TRAIN_FILE, dtype={col: 'float32' for col in self.feature_columns})
        test_df = pd.read_csv(Config.TEST_FILE, dtype={col: 'float32' for col in self.feature_columns})
        
        print(f"Train 데이터 형태: {train_df.shape}")
        print(f"Test 데이터 형태: {test_df.shape}")
        
        # 타겟 컬럼은 정확도를 위해 int32 유지
        if Config.TARGET_COLUMN in train_df.columns:
            train_df[Config.TARGET_COLUMN] = train_df[Config.TARGET_COLUMN].astype('int32')
        
        train_quality = check_data_quality(train_df, self.feature_columns)
        test_quality = check_data_quality(test_df, self.feature_columns)
        
        if not (train_quality and test_quality):
            print("데이터 품질 문제 발견, 추가 전처리 진행")
            train_df, test_df = self._handle_data_issues(train_df, test_df)
        
        return train_df, test_df
    
    def _handle_data_issues(self, train_df, test_df):
        """데이터 이슈 통합 처리"""
        print("데이터 이슈 처리 중...")
        
        # 결측치와 무한값을 한 번에 처리
        for col in self.feature_columns:
            # 무한값을 NaN으로 변환
            train_df[col] = train_df[col].replace([np.inf, -np.inf], np.nan)
            test_df[col] = test_df[col].replace([np.inf, -np.inf], np.nan)
            
            # 결측치가 있는 경우
            if train_df[col].isnull().sum() > 0:
                if train_df[col].std() > 0.1:
                    fill_val = train_df[col].median()
                else:
                    fill_val = train_df[col].mean()
                
                # 결측치가 여전히 발생하면 0으로 대체
                if pd.isna(fill_val):
                    fill_val = 0.0
                    
                train_df[col].fillna(fill_val, inplace=True)
                test_df[col].fillna(fill_val, inplace=True)
        
        return train_df, test_df
    
    def _calculate_sensor_groups(self, train_df):
        """센서 데이터 상관관계 기반 그룹화"""
        print("센서 상관관계 분석 중...")
        
        # 메모리 효율적 상관관계 계산
        corr_matrix = train_df[self.feature_columns].corr().abs()
        
        # 고상관 센서 쌍 식별
        high_corr_pairs = []
        n_features = len(self.feature_columns)
        
        for i in range(n_features):
            for j in range(i+1, n_features):
                corr_val = corr_matrix.iloc[i, j]
                if corr_val > 0.8:
                    high_corr_pairs.append((
                        self.feature_columns[i], 
                        self.feature_columns[j],
                        corr_val
                    ))
        
        # 상위 10개만 사용하여 메모리 절약
        self.high_corr_pairs = sorted(high_corr_pairs, key=lambda x: x[2], reverse=True)[:10]
        print(f"고상관 센서 쌍: {len(self.high_corr_pairs)}개")
        
        # 분산 기반 센서 분류
        variances = train_df[self.feature_columns].var()
        high_var_threshold = variances.quantile(0.7)
        low_var_threshold = variances.quantile(0.3)
        
        self.high_var_sensors = variances[variances > high_var_threshold].index.tolist()
        self.low_var_sensors = variances[variances <= low_var_threshold].index.tolist()
        
        print(f"고분산 센서: {len(self.high_var_sensors)}개")
        print(f"저분산 센서: {len(self.low_var_sensors)}개")
        
        # 메모리 정리
        del corr_matrix, variances
        gc.collect()
    
    @timer
    def feature_engineering(self, train_df, test_df):
        """피처 엔지니어링"""
        print("=== 피처 엔지니어링 시작 ===")
        
        # 메모리 효율적 복사
        X_train = train_df[self.feature_columns].copy()
        X_test = test_df[self.feature_columns].copy()
        y_train = train_df[Config.TARGET_COLUMN].copy()
        
        # 센서 그룹 분석
        self._calculate_sensor_groups(train_df)
        
        # 통계 피처 생성
        X_train, X_test = self._create_statistical_features(X_train, X_test)
        
        # 상호작용 피처 생성
        X_train, X_test = self._create_interaction_features(X_train, X_test)
        
        # 최종 데이터 정리
        X_train, X_test = self._final_cleanup(X_train, X_test)
        
        return X_train, X_test, y_train
    
    def _create_statistical_features(self, X_train, X_test):
        """통계 피처 생성"""
        print("통계 피처 생성 중...")
        
        # 기본 통계량
        feature_arrays = [X_train[self.feature_columns].values, X_test[self.feature_columns].values]
        
        for i, (X, data_name) in enumerate([(X_train, 'train'), (X_test, 'test')]):
            arr = feature_arrays[i]
            
            # 벡터화된 연산으로 성능 향상
            X['sensor_mean'] = np.mean(arr, axis=1).astype('float32')
            X['sensor_std'] = np.std(arr, axis=1).astype('float32')
            X['sensor_range'] = (np.max(arr, axis=1) - np.min(arr, axis=1)).astype('float32')
            
            # 분산 그룹별 통계
            if len(self.high_var_sensors) > 0:
                high_var_data = X[self.high_var_sensors].values
                X['high_var_mean'] = np.mean(high_var_data, axis=1).astype('float32')
                
            if len(self.low_var_sensors) > 0:
                low_var_data = X[self.low_var_sensors].values
                X['low_var_mean'] = np.mean(low_var_data, axis=1).astype('float32')
        
        return X_train, X_test
    
    def _create_interaction_features(self, X_train, X_test):
        """상호작용 피처 생성"""
        print("상호작용 피처 생성 중...")
        
        # 상위 5개 고상관 센서 쌍만 사용
        for s1, s2, corr in self.high_corr_pairs[:5]:
            # 안전한 나눗셈을 위한 보정
            eps = 1e-8
            
            # 비율 피처
            ratio_name = f'ratio_{s1}_{s2}'
            X_train[ratio_name] = (X_train[s1] / (X_train[s2].abs() + eps)).astype('float32')
            X_test[ratio_name] = (X_test[s1] / (X_test[s2].abs() + eps)).astype('float32')
            
            # 차분 피처
            diff_name = f'diff_{s1}_{s2}'
            X_train[diff_name] = (X_train[s1] - X_train[s2]).astype('float32')
            X_test[diff_name] = (X_test[s1] - X_test[s2]).astype('float32')
        
        return X_train, X_test
    
    def _final_cleanup(self, X_train, X_test):
        """최종 데이터 정리"""
        print("최종 데이터 정리 중...")
        
        # 무한값과 NaN을 한 번에 처리
        for df in [X_train, X_test]:
            # 무한값을 NaN으로 변환
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            # NaN을 0으로 대체 (이미 검증된 데이터이므로 단순 처리)
            df.fillna(0, inplace=True)
            
            # 데이터 타입 최적화
            for col in df.select_dtypes(include=['float64']).columns:
                df[col] = df[col].astype('float32')
        
        print(f"정리 후 훈련 데이터 NaN: {X_train.isna().sum().sum()}")
        print(f"정리 후 테스트 데이터 NaN: {X_test.isna().sum().sum()}")
        print(f"생성된 피처 수: {X_train.shape[1]}")
        
        # 메모리 정리
        gc.collect()
        
        return X_train, X_test
    
    @timer
    def scale_features(self, X_train, X_test, method='robust'):
        """피처 스케일링"""
        print(f"=== 피처 스케일링 시작 ({method}) ===")
        
        if method == 'robust':
            self.scaler = RobustScaler()
        elif method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'quantile':
            self.scaler = QuantileTransformer(output_distribution='normal', random_state=Config.RANDOM_STATE)
        else:
            self.scaler = RobustScaler()
        
        # 메모리 효율적 스케일링
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # DataFrame 변환 (인덱스 유지)
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
        
        k = min(k, X_train.shape[1])  # 피처 수를 초과하지 않도록 제한
        
        print(f"=== 피처 선택 시작 ({method}, k={k}) ===")
        print(f"원본 피처 개수: {X_train.shape[1]}")
        
        if method == 'mutual_info':
            selector = SelectKBest(mutual_info_classif, k=k)
        elif method == 'f_classif':
            selector = SelectKBest(f_classif, k=k)
        else:
            selector = SelectKBest(mutual_info_classif, k=k)
        
        self.feature_selector = selector
        
        # 피처 선택 수행
        X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
        X_test_selected = self.feature_selector.transform(X_test)
        
        selected_mask = self.feature_selector.get_support()
        self.selected_features = X_train.columns[selected_mask].tolist()
        
        print(f"선택된 피처 개수: {len(self.selected_features)}")
        
        # DataFrame 변환
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
        
        save_joblib(self.feature_selector, Config.FEATURE_SELECTOR_FILE)
        
        # 메모리 정리
        gc.collect()
        
        return X_train_selected, X_test_selected
    
    @timer
    def apply_pca(self, X_train, X_test, n_components=0.95):
        """PCA 차원 축소"""
        print(f"=== PCA 적용 시작 (n_components={n_components}) ===")
        
        self.pca = PCA(n_components=n_components, random_state=Config.RANDOM_STATE)
        
        X_train_pca = self.pca.fit_transform(X_train)
        X_test_pca = self.pca.transform(X_test)
        
        n_components_actual = X_train_pca.shape[1]
        explained_variance = self.pca.explained_variance_ratio_.sum()
        
        print(f"실제 컴포넌트 수: {n_components_actual}")
        print(f"설명 분산 비율: {explained_variance:.4f}")
        
        pca_columns = [f'pca_{i}' for i in range(n_components_actual)]
        
        X_train_pca = pd.DataFrame(
            X_train_pca,
            columns=pca_columns,
            index=X_train.index,
            dtype='float32'
        )
        X_test_pca = pd.DataFrame(
            X_test_pca,
            columns=pca_columns,
            index=X_test.index,
            dtype='float32'
        )
        
        save_joblib(self.pca, Config.PCA_FILE)
        
        return X_train_pca, X_test_pca
    
    def get_processed_data(self, use_feature_selection=True, use_pca=False, scaling_method='robust'):
        """전체 전처리 파이프라인 실행"""
        print("=== 전체 데이터 전처리 파이프라인 시작 ===")
        
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
            
            # 5. PCA (선택적)
            if use_pca:
                X_train, X_test = self.apply_pca(X_train, X_test, n_components=0.95)
            
            # 6. 최종 검증 및 정리
            print("=== 최종 데이터 검증 ===")
            
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
            
            # ID 컬럼 반환을 위한 원본 로드 (메모리 효율적)
            train_ids = pd.read_csv(Config.TRAIN_FILE, usecols=[Config.ID_COLUMN])[Config.ID_COLUMN]
            test_ids = pd.read_csv(Config.TEST_FILE, usecols=[Config.ID_COLUMN])[Config.ID_COLUMN]
            
            return X_train, X_test, y_train, train_ids, test_ids
            
        except Exception as e:
            print(f"데이터 전처리 중 오류 발생: {e}")
            # 메모리 정리
            gc.collect()
            raise
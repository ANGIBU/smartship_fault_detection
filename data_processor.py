# data_processor.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler, QuantileTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
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
        
        train_df = pd.read_csv(Config.TRAIN_FILE)
        test_df = pd.read_csv(Config.TEST_FILE)
        
        print(f"Train 데이터 형태: {train_df.shape}")
        print(f"Test 데이터 형태: {test_df.shape}")
        
        # float64 정밀도 유지 (성능 우선)
        for col in self.feature_columns:
            if col in train_df.columns:
                train_df[col] = train_df[col].astype('float64')
            if col in test_df.columns:
                test_df[col] = test_df[col].astype('float64')
        
        train_quality = check_data_quality(train_df, self.feature_columns)
        test_quality = check_data_quality(test_df, self.feature_columns)
        
        if not (train_quality and test_quality):
            print("데이터 품질 문제 발견, 추가 전처리 진행")
            train_df, test_df = self._handle_data_issues(train_df, test_df)
        
        return train_df, test_df
    
    def _handle_data_issues(self, train_df, test_df):
        """데이터 이슈 처리"""
        print("데이터 이슈 처리 중...")
        
        for col in self.feature_columns:
            if train_df[col].isnull().sum() > 0:
                # 센서 특성을 고려한 결측치 처리
                if train_df[col].std() > 0.1:  # 고분산 센서
                    fill_val = train_df[col].median()
                else:  # 저분산 센서
                    fill_val = train_df[col].mean()
                    
                train_df[col].fillna(fill_val, inplace=True)
                test_df[col].fillna(fill_val, inplace=True)
                print(f"{col} 결측치 처리: {fill_val:.6f}")
        
        # 무한값을 센서별 99.9% 분위수로 대체
        for col in self.feature_columns:
            train_df[col] = train_df[col].replace([np.inf, -np.inf], np.nan)
            test_df[col] = test_df[col].replace([np.inf, -np.inf], np.nan)
            
            if train_df[col].isnull().sum() > 0:
                clip_val = train_df[col].quantile(0.999)
                train_df[col].fillna(clip_val, inplace=True)
                test_df[col].fillna(clip_val, inplace=True)
        
        return train_df, test_df
    
    def _calculate_sensor_groups(self, train_df):
        """센서 데이터 상관관계 기반 그룹화"""
        print("센서 상관관계 분석 중...")
        
        corr_matrix = train_df[self.feature_columns].corr().abs()
        
        # 고상관 센서 쌍 식별 (0.8 이상)
        high_corr_pairs = []
        for i in range(len(self.feature_columns)):
            for j in range(i+1, len(self.feature_columns)):
                if corr_matrix.iloc[i, j] > 0.8:
                    high_corr_pairs.append((
                        self.feature_columns[i], 
                        self.feature_columns[j],
                        corr_matrix.iloc[i, j]
                    ))
        
        self.high_corr_pairs = high_corr_pairs[:10]  # 상위 10개만 사용
        print(f"고상관 센서 쌍: {len(self.high_corr_pairs)}개")
        
        # 분산 기반 센서 분류
        variances = train_df[self.feature_columns].var()
        self.high_var_sensors = variances[variances > variances.quantile(0.7)].index.tolist()
        self.low_var_sensors = variances[variances <= variances.quantile(0.3)].index.tolist()
        
        print(f"고분산 센서: {len(self.high_var_sensors)}개")
        print(f"저분산 센서: {len(self.low_var_sensors)}개")
    
    @timer
    def feature_engineering(self, train_df, test_df):
        """피처 엔지니어링 - 단순화 및 정밀화"""
        print("=== 피처 엔지니어링 시작 ===")
        
        X_train = train_df[self.feature_columns].copy()
        X_test = test_df[self.feature_columns].copy()
        y_train = train_df[Config.TARGET_COLUMN]
        
        # 센서 그룹 분석
        self._calculate_sensor_groups(train_df)
        
        # 선별된 통계 피처만 생성
        X_train, X_test = self._create_selective_features(X_train, X_test)
        
        # 센서 간 상호작용 피처 (고상관 쌍만)
        X_train, X_test = self._create_interaction_features(X_train, X_test)
        
        # 최종 정리
        X_train, X_test = self._final_cleanup(X_train, X_test)
        
        return X_train, X_test, y_train
    
    def _create_selective_features(self, X_train, X_test):
        """선별적 통계 피처 생성"""
        print("선별적 통계 피처 생성 중...")
        
        # 핵심 통계량만 생성 (중복 최소화)
        X_train['sensor_mean'] = X_train[self.feature_columns].mean(axis=1)
        X_train['sensor_std'] = X_train[self.feature_columns].std(axis=1)
        X_train['sensor_range'] = X_train[self.feature_columns].max(axis=1) - X_train[self.feature_columns].min(axis=1)
        X_train['sensor_skew'] = X_train[self.feature_columns].skew(axis=1)
        
        X_test['sensor_mean'] = X_test[self.feature_columns].mean(axis=1)
        X_test['sensor_std'] = X_test[self.feature_columns].std(axis=1)
        X_test['sensor_range'] = X_test[self.feature_columns].max(axis=1) - X_test[self.feature_columns].min(axis=1)
        X_test['sensor_skew'] = X_test[self.feature_columns].skew(axis=1)
        
        # 분산 그룹별 통계
        if len(self.high_var_sensors) > 0:
            X_train['high_var_mean'] = X_train[self.high_var_sensors].mean(axis=1)
            X_test['high_var_mean'] = X_test[self.high_var_sensors].mean(axis=1)
            
        if len(self.low_var_sensors) > 0:
            X_train['low_var_mean'] = X_train[self.low_var_sensors].mean(axis=1)
            X_test['low_var_mean'] = X_test[self.low_var_sensors].mean(axis=1)
        
        return X_train, X_test
    
    def _create_interaction_features(self, X_train, X_test):
        """상관관계 기반 상호작용 피처"""
        print("상호작용 피처 생성 중...")
        
        # 고상관 센서 쌍의 상호작용만 생성
        for s1, s2, corr in self.high_corr_pairs[:5]:  # 상위 5개만
            # 비율 피처 (분모 보정 정밀화)
            ratio_name = f'ratio_{s1}_{s2}'
            denominator = X_train[s2].abs() + 1e-10  # 더 안전한 보정값
            X_train[ratio_name] = X_train[s1] / denominator
            
            denominator_test = X_test[s2].abs() + 1e-10
            X_test[ratio_name] = X_test[s1] / denominator_test
            
            # 차분 피처
            diff_name = f'diff_{s1}_{s2}'
            X_train[diff_name] = X_train[s1] - X_train[s2]
            X_test[diff_name] = X_test[s1] - X_test[s2]
        
        return X_train, X_test
    
    def _final_cleanup(self, X_train, X_test):
        """최종 데이터 정리"""
        print("최종 데이터 정리 중...")
        
        # 무한값 및 NaN 처리
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        
        # 훈련 데이터 기준 결측치 처리
        for col in X_train.columns:
            if X_train[col].isna().any():
                if X_train[col].dtype in ['float64', 'float32']:
                    fill_val = X_train[col].median()
                else:
                    fill_val = 0
                    
                if pd.isna(fill_val):
                    fill_val = 0
                    
                X_train[col].fillna(fill_val, inplace=True)
                X_test[col].fillna(fill_val, inplace=True)
        
        print(f"정리 후 훈련 데이터 NaN: {X_train.isna().sum().sum()}")
        print(f"정리 후 테스트 데이터 NaN: {X_test.isna().sum().sum()}")
        print(f"생성된 피처 수: {X_train.shape[1]}")
        
        return X_train, X_test
    
    @timer
    def scale_features(self, X_train, X_test, method='robust'):
        """피처 스케일링 - 다중 방법 지원"""
        print(f"=== 피처 스케일링 시작 ({method}) ===")
        
        if method == 'robust':
            self.scaler = RobustScaler()
        elif method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'quantile':
            self.scaler = QuantileTransformer(output_distribution='normal', random_state=Config.RANDOM_STATE)
        else:
            self.scaler = RobustScaler()
        
        # 훈련 데이터만으로 fit
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # DataFrame 변환
        X_train_scaled = pd.DataFrame(
            X_train_scaled, 
            columns=X_train.columns, 
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            X_test_scaled, 
            columns=X_test.columns, 
            index=X_test.index
        )
        
        save_joblib(self.scaler, Config.SCALER_FILE)
        
        return X_train_scaled, X_test_scaled
    
    @timer
    def select_features(self, X_train, X_test, y_train, method='mutual_info', k=None):
        """피처 선택 - 다중 방법 조합"""
        if k is None:
            k = Config.FEATURE_SELECTION_K
        
        print(f"=== 피처 선택 시작 ({method}, k={k}) ===")
        print(f"원본 피처 개수: {X_train.shape[1]}")
        
        if method == 'mutual_info':
            selector = SelectKBest(mutual_info_classif, k=k)
        elif method == 'f_classif':
            selector = SelectKBest(f_classif, k=k)
        else:
            selector = SelectKBest(mutual_info_classif, k=k)
        
        self.feature_selector = selector
        
        # 훈련 데이터만으로 fit
        X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
        X_test_selected = self.feature_selector.transform(X_test)
        
        selected_mask = self.feature_selector.get_support()
        self.selected_features = X_train.columns[selected_mask].tolist()
        
        print(f"선택된 피처 개수: {len(self.selected_features)}")
        
        X_train_selected = pd.DataFrame(
            X_train_selected, 
            columns=self.selected_features, 
            index=X_train.index
        )
        X_test_selected = pd.DataFrame(
            X_test_selected, 
            columns=self.selected_features, 
            index=X_test.index
        )
        
        save_joblib(self.feature_selector, Config.FEATURE_SELECTOR_FILE)
        
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
            index=X_train.index
        )
        X_test_pca = pd.DataFrame(
            X_test_pca,
            columns=pca_columns,
            index=X_test.index
        )
        
        save_joblib(self.pca, Config.PCA_FILE)
        
        return X_train_pca, X_test_pca
    
    def get_processed_data(self, use_feature_selection=True, use_pca=False, scaling_method='robust'):
        """전체 전처리 파이프라인 실행"""
        print("=== 전체 데이터 전처리 파이프라인 시작 ===")
        
        # 1. 데이터 로드
        train_df, test_df = self.load_and_preprocess_data()
        
        # 2. 피처 엔지니어링
        X_train, X_test, y_train = self.feature_engineering(train_df, test_df)
        
        # 3. 스케일링
        X_train, X_test = self.scale_features(X_train, X_test, scaling_method)
        
        # 4. 피처 선택
        if use_feature_selection:
            X_train, X_test = self.select_features(
                X_train, X_test, y_train, 
                method='mutual_info', 
                k=min(Config.FEATURE_SELECTION_K, X_train.shape[1])
            )
        
        # 5. PCA (선택적)
        if use_pca:
            X_train, X_test = self.apply_pca(X_train, X_test, n_components=0.95)
        
        # 6. 최종 검증
        print("=== 최종 데이터 검증 ===")
        final_nan_train = X_train.isna().sum().sum()
        final_nan_test = X_test.isna().sum().sum()
        
        print(f"최종 훈련 데이터 NaN: {final_nan_train}")
        print(f"최종 테스트 데이터 NaN: {final_nan_test}")
        
        if final_nan_train > 0 or final_nan_test > 0:
            X_train = X_train.fillna(0)
            X_test = X_test.fillna(0)
            print("잔여 NaN을 0으로 대체")
        
        print(f"최종 피처 개수: {X_train.shape[1]}")
        print(f"최종 데이터 형태 - 훈련: {X_train.shape}, 테스트: {X_test.shape}")
        
        return X_train, X_test, y_train, train_df[Config.ID_COLUMN], test_df[Config.ID_COLUMN]
# data_processor.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

from config import Config
from utils import timer, check_data_quality, optimize_memory_usage, save_joblib, load_joblib

class DataProcessor:
    def __init__(self):
        self.scaler = None
        self.feature_selector = None
        self.imputer = None
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
        
        train_df = optimize_memory_usage(train_df)
        test_df = optimize_memory_usage(test_df)
        
        train_quality = check_data_quality(train_df, self.feature_columns)
        test_quality = check_data_quality(test_df, self.feature_columns)
        
        if not (train_quality and test_quality):
            print("데이터 품질 문제 발견, 추가 전처리 진행")
            train_df, test_df = self._handle_data_issues(train_df, test_df)
        
        self._calculate_train_stats(train_df)
        
        return train_df, test_df
    
    def _handle_data_issues(self, train_df, test_df):
        """데이터 이슈 처리"""
        print("데이터 이슈 처리 중...")
        
        for col in self.feature_columns:
            if train_df[col].isnull().sum() > 0:
                median_val = train_df[col].median()
                train_df[col].fillna(median_val, inplace=True)
                test_df[col].fillna(median_val, inplace=True)
                print(f"{col} 결측치 {median_val}로 대체")
        
        for col in self.feature_columns:
            train_df[col] = train_df[col].replace([np.inf, -np.inf], np.nan)
            test_df[col] = test_df[col].replace([np.inf, -np.inf], np.nan)
            
            if train_df[col].isnull().sum() > 0:
                median_val = train_df[col].median()
                train_df[col].fillna(median_val, inplace=True)
                test_df[col].fillna(median_val, inplace=True)
                print(f"{col} 무한값 {median_val}로 대체")
        
        return train_df, test_df
    
    def _calculate_train_stats(self, train_df):
        """훈련 데이터 기반 통계 계산"""
        print("훈련 데이터 통계 계산 중...")
        
        for col in self.feature_columns:
            stats = {
                'mean': train_df[col].mean(),
                'std': train_df[col].std(),
                'median': train_df[col].median(),
                'q25': train_df[col].quantile(0.25),
                'q75': train_df[col].quantile(0.75)
            }
            self.train_stats[col] = stats
        
        self.high_variance_features = []
        self.low_variance_features = []
        
        for col in self.feature_columns:
            std_val = self.train_stats[col]['std']
            if std_val > 0.1:
                self.high_variance_features.append(col)
            else:
                self.low_variance_features.append(col)
        
        print(f"고분산 피처: {len(self.high_variance_features)}개")
        print(f"저분산 피처: {len(self.low_variance_features)}개")
    
    @timer
    def feature_engineering(self, train_df, test_df):
        """피처 엔지니어링 - 리키지 방지"""
        print("=== 피처 엔지니어링 시작 ===")
        
        X_train = train_df[self.feature_columns].copy()
        X_test = test_df[self.feature_columns].copy()
        y_train = train_df[Config.TARGET_COLUMN]
        
        # 훈련 데이터만으로 통계 피처 생성
        X_train, X_test = self._create_statistical_features(X_train, X_test)
        
        # 비율 및 차분 피처 생성
        X_train, X_test = self._create_ratio_features(X_train, X_test)
        
        # 센서 그룹별 피처 생성
        X_train, X_test = self._create_group_features(X_train, X_test)
        
        # 데이터 검증 및 정리
        X_train, X_test = self._final_data_cleanup(X_train, X_test)
        
        return X_train, X_test, y_train
    
    def _create_statistical_features(self, X_train, X_test):
        """통계적 피처 생성 - 리키지 방지"""
        print("통계적 피처 생성 중...")
        
        # 훈련 데이터 기준 통계량 계산
        train_mean = X_train[self.feature_columns].mean(axis=1)
        train_std = X_train[self.feature_columns].std(axis=1)
        train_max = X_train[self.feature_columns].max(axis=1)
        train_min = X_train[self.feature_columns].min(axis=1)
        train_median = X_train[self.feature_columns].median(axis=1)
        
        # 동일한 방식으로 테스트 데이터 계산
        test_mean = X_test[self.feature_columns].mean(axis=1)
        test_std = X_test[self.feature_columns].std(axis=1)
        test_max = X_test[self.feature_columns].max(axis=1)
        test_min = X_test[self.feature_columns].min(axis=1)
        test_median = X_test[self.feature_columns].median(axis=1)
        
        # 피처 추가
        X_train['feature_mean'] = train_mean
        X_train['feature_std'] = train_std
        X_train['feature_max'] = train_max
        X_train['feature_min'] = train_min
        X_train['feature_median'] = train_median
        X_train['feature_range'] = train_max - train_min
        
        X_test['feature_mean'] = test_mean
        X_test['feature_std'] = test_std
        X_test['feature_max'] = test_max
        X_test['feature_min'] = test_min
        X_test['feature_median'] = test_median
        X_test['feature_range'] = test_max - test_min
        
        # 분위수 피처
        for q in [0.25, 0.75]:
            train_q = X_train[self.feature_columns].quantile(q, axis=1)
            test_q = X_test[self.feature_columns].quantile(q, axis=1)
            
            X_train[f'feature_q{int(q*100)}'] = train_q
            X_test[f'feature_q{int(q*100)}'] = test_q
        
        return X_train, X_test
    
    def _create_ratio_features(self, X_train, X_test):
        """비율 피처 생성"""
        print("비율 피처 생성 중...")
        
        # 주요 센서 간 비율
        sensor_pairs = [
            ('X_01', 'X_02'), ('X_03', 'X_04'), ('X_05', 'X_06'),
            ('X_07', 'X_08'), ('X_09', 'X_10')
        ]
        
        for s1, s2 in sensor_pairs:
            if s1 in X_train.columns and s2 in X_train.columns:
                # 안전한 나눗셈
                denominator_train = X_train[s2].abs() + 1e-8
                denominator_test = X_test[s2].abs() + 1e-8
                
                ratio_name = f'ratio_{s1}_{s2}'
                X_train[ratio_name] = X_train[s1] / denominator_train
                X_test[ratio_name] = X_test[s1] / denominator_test
        
        # 차분 피처
        for i in range(1, 6):
            col1 = f'X_{i:02d}'
            col2 = f'X_{i+1:02d}' if i < 5 else f'X_{i-1:02d}'
            
            if col1 in X_train.columns and col2 in X_train.columns:
                diff_name = f'diff_{col1}_{col2}'
                X_train[diff_name] = X_train[col1] - X_train[col2]
                X_test[diff_name] = X_test[col1] - X_test[col2]
        
        return X_train, X_test
    
    def _create_group_features(self, X_train, X_test):
        """그룹별 피처 생성"""
        print("그룹별 피처 생성 중...")
        
        # 센서 그룹 (10개씩)
        group_size = 10
        for i in range(0, len(self.feature_columns), group_size):
            end_idx = min(i + group_size, len(self.feature_columns))
            group_features = self.feature_columns[i:end_idx]
            group_name = f'group_{i//group_size + 1}'
            
            if len(group_features) >= 2:
                X_train[f'{group_name}_mean'] = X_train[group_features].mean(axis=1)
                X_train[f'{group_name}_std'] = X_train[group_features].std(axis=1)
                
                X_test[f'{group_name}_mean'] = X_test[group_features].mean(axis=1)
                X_test[f'{group_name}_std'] = X_test[group_features].std(axis=1)
        
        return X_train, X_test
    
    def _final_data_cleanup(self, X_train, X_test):
        """최종 데이터 정리"""
        print("최종 데이터 검증 및 정리 중...")
        
        print(f"정리 전 - 훈련 데이터 NaN 개수: {X_train.isna().sum().sum()}")
        print(f"정리 전 - 테스트 데이터 NaN 개수: {X_test.isna().sum().sum()}")
        
        # 무한값을 NaN으로 변환
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        
        # 훈련 데이터 기준으로 결측치 처리
        for col in X_train.columns:
            if X_train[col].isna().any():
                median_val = X_train[col].median()
                if pd.isna(median_val):
                    median_val = 0
                
                X_train[col].fillna(median_val, inplace=True)
                X_test[col].fillna(median_val, inplace=True)
        
        print(f"정리 후 - 훈련 데이터 NaN 개수: {X_train.isna().sum().sum()}")
        print(f"정리 후 - 테스트 데이터 NaN 개수: {X_test.isna().sum().sum()}")
        
        return X_train, X_test
    
    @timer
    def scale_features(self, X_train, X_test, method='robust'):
        """피처 스케일링 - 리키지 방지"""
        print(f"=== 피처 스케일링 시작 ({method}) ===")
        
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
        
        # 결측치 최종 처리
        if X_train_scaled.isna().sum().sum() > 0 or X_test_scaled.isna().sum().sum() > 0:
            print("스케일링 후 NaN 값 처리 중...")
            imputer = SimpleImputer(strategy='median')
            
            X_train_scaled = pd.DataFrame(
                imputer.fit_transform(X_train_scaled),
                columns=X_train_scaled.columns,
                index=X_train_scaled.index
            )
            X_test_scaled = pd.DataFrame(
                imputer.transform(X_test_scaled),
                columns=X_test_scaled.columns,
                index=X_test_scaled.index
            )
        
        save_joblib(self.scaler, Config.SCALER_FILE)
        
        return X_train_scaled, X_test_scaled
    
    @timer
    def select_features(self, X_train, X_test, y_train, method='mutual_info', k=None):
        """피처 선택 - 리키지 방지"""
        if k is None:
            k = Config.FEATURE_SELECTION_K
        
        print(f"=== 피처 선택 시작 ({method}, k={k}) ===")
        print(f"원본 피처 개수: {X_train.shape[1]}")
        
        # 훈련 데이터만으로 피처 선택
        if method == 'mutual_info':
            self.feature_selector = SelectKBest(mutual_info_classif, k=k)
        else:
            from sklearn.feature_selection import f_classif
            self.feature_selector = SelectKBest(f_classif, k=k)
        
        X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
        X_test_selected = self.feature_selector.transform(X_test)
        
        # 선택된 피처명 저장
        selected_mask = self.feature_selector.get_support()
        self.selected_features = X_train.columns[selected_mask].tolist()
        
        print(f"선택된 피처 개수: {len(self.selected_features)}")
        
        # DataFrame 변환
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
    
    def get_processed_data(self, use_feature_selection=True):
        """전체 전처리 파이프라인 실행"""
        print("=== 전체 데이터 전처리 파이프라인 시작 ===")
        
        # 데이터 로드 및 기본 전처리
        train_df, test_df = self.load_and_preprocess_data()
        
        # 피처 엔지니어링
        X_train, X_test, y_train = self.feature_engineering(train_df, test_df)
        
        # 스케일링
        X_train, X_test = self.scale_features(X_train, X_test, method='robust')
        
        # 피처 선택
        if use_feature_selection:
            X_train, X_test = self.select_features(
                X_train, X_test, y_train, 
                method='mutual_info', 
                k=Config.FEATURE_SELECTION_K
            )
        
        # 최종 검증
        print("=== 최종 데이터 검증 ===")
        final_nan_train = X_train.isna().sum().sum()
        final_nan_test = X_test.isna().sum().sum()
        final_inf_train = np.isinf(X_train.select_dtypes(include=[np.number])).sum().sum()
        final_inf_test = np.isinf(X_test.select_dtypes(include=[np.number])).sum().sum()
        
        print(f"최종 훈련 데이터 NaN 개수: {final_nan_train}")
        print(f"최종 테스트 데이터 NaN 개수: {final_nan_test}")
        print(f"최종 훈련 데이터 무한값 개수: {final_inf_train}")
        print(f"최종 테스트 데이터 무한값 개수: {final_inf_test}")
        
        if final_nan_train > 0 or final_nan_test > 0 or final_inf_train > 0 or final_inf_test > 0:
            print("최종 단계에서 문제 값 발견, 정리...")
            X_train = X_train.replace([np.inf, -np.inf, np.nan], 0)
            X_test = X_test.replace([np.inf, -np.inf, np.nan], 0)
            print("모든 문제 값을 0으로 대체 완료")
        
        print("=== 전처리 완료 ===")
        print(f"최종 피처 개수: {X_train.shape[1]}")
        print(f"최종 데이터 형태 - 훈련: {X_train.shape}, 테스트: {X_test.shape}")
        
        return X_train, X_test, y_train, train_df[Config.ID_COLUMN], test_df[Config.ID_COLUMN]
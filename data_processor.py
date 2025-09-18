# data_processor.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

from config import Config
from utils import timer, check_data_quality, optimize_memory_usage, save_joblib

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
        
        return train_df, test_df
    
    def _handle_data_issues(self, train_df, test_df):
        """데이터 이슈 처리"""
        print("데이터 이슈 처리 중...")
        
        # 훈련 데이터만으로 결측치 처리
        for col in self.feature_columns:
            if train_df[col].isnull().sum() > 0:
                median_val = train_df[col].median()
                train_df[col].fillna(median_val, inplace=True)
                test_df[col].fillna(median_val, inplace=True)
                print(f"{col} 결측치 {median_val}로 대체")
        
        # 무한값 처리
        for col in self.feature_columns:
            train_df[col] = train_df[col].replace([np.inf, -np.inf], np.nan)
            test_df[col] = test_df[col].replace([np.inf, -np.inf], np.nan)
            
            if train_df[col].isnull().sum() > 0:
                median_val = train_df[col].median()
                train_df[col].fillna(median_val, inplace=True)
                test_df[col].fillna(median_val, inplace=True)
        
        return train_df, test_df
    
    def _calculate_train_stats(self, train_df):
        """훈련 데이터만으로 통계 계산"""
        print("훈련 데이터 통계 계산 중...")
        
        for col in self.feature_columns:
            stats = {
                'mean': train_df[col].mean(),
                'std': train_df[col].std(),
                'median': train_df[col].median(),
                'q25': train_df[col].quantile(0.25),
                'q75': train_df[col].quantile(0.75),
                'min': train_df[col].min(),
                'max': train_df[col].max()
            }
            self.train_stats[col] = stats
        
        # 분산 기반 피처 그룹화
        high_var = []
        low_var = []
        
        for col in self.feature_columns:
            var_val = self.train_stats[col]['std']
            if var_val > 0.1:
                high_var.append(col)
            else:
                low_var.append(col)
        
        self.high_variance_features = high_var
        self.low_variance_features = low_var
        
        print(f"고분산 피처: {len(high_var)}개")
        print(f"저분산 피처: {len(low_var)}개")
    
    @timer
    def feature_engineering(self, train_df, test_df):
        """피처 엔지니어링"""
        print("=== 피처 엔지니어링 시작 ===")
        
        X_train = train_df[self.feature_columns].copy()
        X_test = test_df[self.feature_columns].copy()
        y_train = train_df[Config.TARGET_COLUMN]
        
        # 훈련 데이터 통계 계산
        self._calculate_train_stats(train_df)
        
        # 통계 피처 생성
        X_train, X_test = self._create_statistical_features(X_train, X_test)
        
        # 센서 상호작용 피처 생성
        X_train, X_test = self._create_interaction_features(X_train, X_test)
        
        # 그룹 피처 생성
        X_train, X_test = self._create_group_features(X_train, X_test)
        
        # 최종 정리
        X_train, X_test = self._final_cleanup(X_train, X_test)
        
        return X_train, X_test, y_train
    
    def _create_statistical_features(self, X_train, X_test):
        """통계적 피처 생성"""
        print("통계적 피처 생성 중...")
        
        # 기본 통계량
        X_train['stat_mean'] = X_train[self.feature_columns].mean(axis=1)
        X_train['stat_std'] = X_train[self.feature_columns].std(axis=1)
        X_train['stat_max'] = X_train[self.feature_columns].max(axis=1)
        X_train['stat_min'] = X_train[self.feature_columns].min(axis=1)
        X_train['stat_range'] = X_train['stat_max'] - X_train['stat_min']
        X_train['stat_median'] = X_train[self.feature_columns].median(axis=1)
        
        X_test['stat_mean'] = X_test[self.feature_columns].mean(axis=1)
        X_test['stat_std'] = X_test[self.feature_columns].std(axis=1)
        X_test['stat_max'] = X_test[self.feature_columns].max(axis=1)
        X_test['stat_min'] = X_test[self.feature_columns].min(axis=1)
        X_test['stat_range'] = X_test['stat_max'] - X_test['stat_min']
        X_test['stat_median'] = X_test[self.feature_columns].median(axis=1)
        
        # 분위수
        for q in [0.25, 0.75]:
            q_name = f'stat_q{int(q*100)}'
            X_train[q_name] = X_train[self.feature_columns].quantile(q, axis=1)
            X_test[q_name] = X_test[self.feature_columns].quantile(q, axis=1)
        
        return X_train, X_test
    
    def _create_interaction_features(self, X_train, X_test):
        """센서 상호작용 피처 생성"""
        print("센서 상호작용 피처 생성 중...")
        
        # 주요 센서 간 비율
        pairs = [
            ('X_01', 'X_02'), ('X_03', 'X_04'), ('X_05', 'X_06'),
            ('X_07', 'X_08'), ('X_09', 'X_10'), ('X_11', 'X_12')
        ]
        
        for s1, s2 in pairs:
            if s1 in X_train.columns and s2 in X_train.columns:
                # 비율 피처
                ratio_name = f'ratio_{s1}_{s2}'
                X_train[ratio_name] = X_train[s1] / (X_train[s2].abs() + 1e-8)
                X_test[ratio_name] = X_test[s1] / (X_test[s2].abs() + 1e-8)
                
                # 차분 피처
                diff_name = f'diff_{s1}_{s2}'
                X_train[diff_name] = X_train[s1] - X_train[s2]
                X_test[diff_name] = X_test[s1] - X_test[s2]
                
                # 곱셈 피처
                mult_name = f'mult_{s1}_{s2}'
                X_train[mult_name] = X_train[s1] * X_train[s2]
                X_test[mult_name] = X_test[s1] * X_test[s2]
        
        # 고분산 피처들 간 상호작용
        if len(self.high_variance_features) >= 2:
            for i in range(min(3, len(self.high_variance_features))):
                for j in range(i+1, min(4, len(self.high_variance_features))):
                    f1, f2 = self.high_variance_features[i], self.high_variance_features[j]
                    
                    interact_name = f'interact_{f1}_{f2}'
                    X_train[interact_name] = X_train[f1] * X_train[f2]
                    X_test[interact_name] = X_test[f1] * X_test[f2]
        
        return X_train, X_test
    
    def _create_group_features(self, X_train, X_test):
        """그룹별 피처 생성"""
        print("그룹별 피처 생성 중...")
        
        # 10개씩 그룹화
        group_size = 10
        for i in range(0, len(self.feature_columns), group_size):
            end_idx = min(i + group_size, len(self.feature_columns))
            group_features = self.feature_columns[i:end_idx]
            group_id = i // group_size + 1
            
            # 그룹 통계
            X_train[f'group{group_id}_mean'] = X_train[group_features].mean(axis=1)
            X_train[f'group{group_id}_std'] = X_train[group_features].std(axis=1)
            X_train[f'group{group_id}_max'] = X_train[group_features].max(axis=1)
            X_train[f'group{group_id}_min'] = X_train[group_features].min(axis=1)
            
            X_test[f'group{group_id}_mean'] = X_test[group_features].mean(axis=1)
            X_test[f'group{group_id}_std'] = X_test[group_features].std(axis=1)
            X_test[f'group{group_id}_max'] = X_test[group_features].max(axis=1)
            X_test[f'group{group_id}_min'] = X_test[group_features].min(axis=1)
        
        return X_train, X_test
    
    def _final_cleanup(self, X_train, X_test):
        """최종 데이터 정리"""
        print("최종 데이터 정리 중...")
        
        # 무한값 및 NaN 처리
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        
        # 훈련 데이터 기준으로 결측치 처리
        for col in X_train.columns:
            if X_train[col].isna().any():
                fill_val = X_train[col].median()
                if pd.isna(fill_val):
                    fill_val = 0
                X_train[col].fillna(fill_val, inplace=True)
                X_test[col].fillna(fill_val, inplace=True)
        
        print(f"정리 후 훈련 데이터 NaN: {X_train.isna().sum().sum()}")
        print(f"정리 후 테스트 데이터 NaN: {X_test.isna().sum().sum()}")
        
        return X_train, X_test
    
    @timer
    def scale_features(self, X_train, X_test):
        """피처 스케일링"""
        print("=== 피처 스케일링 시작 ===")
        
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
        """피처 선택"""
        if k is None:
            k = Config.FEATURE_SELECTION_K
        
        print(f"=== 피처 선택 시작 ({method}, k={k}) ===")
        print(f"원본 피처 개수: {X_train.shape[1]}")
        
        if method == 'mutual_info':
            self.feature_selector = SelectKBest(mutual_info_classif, k=k)
        else:
            from sklearn.feature_selection import f_classif
            self.feature_selector = SelectKBest(f_classif, k=k)
        
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
    
    def get_processed_data(self, use_feature_selection=True):
        """전체 전처리 파이프라인 실행"""
        print("=== 전체 데이터 전처리 파이프라인 시작 ===")
        
        # 1. 데이터 로드
        train_df, test_df = self.load_and_preprocess_data()
        
        # 2. 피처 엔지니어링
        X_train, X_test, y_train = self.feature_engineering(train_df, test_df)
        
        # 3. 스케일링
        X_train, X_test = self.scale_features(X_train, X_test)
        
        # 4. 피처 선택
        if use_feature_selection:
            X_train, X_test = self.select_features(
                X_train, X_test, y_train, 
                method='mutual_info', 
                k=Config.FEATURE_SELECTION_K
            )
        
        # 5. 최종 검증
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
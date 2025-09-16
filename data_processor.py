# data_processor.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel, RFECV
import warnings
warnings.filterwarnings('ignore')

from config import Config
from utils import timer, check_data_quality, optimize_memory_usage, save_joblib, load_joblib

class DataProcessor:
    def __init__(self):
        self.scaler = None
        self.feature_selector = None
        self.pca = None
        self.svd = None
        self.feature_columns = Config.FEATURE_COLUMNS
        self.selected_features = None
        self.feature_stats = {}
        
    @timer
    def load_and_preprocess_data(self):
        """데이터 로드 및 전처리"""
        print("=== 데이터 로드 및 전처리 시작 ===")
        
        # 데이터 로드
        train_df = pd.read_csv(Config.TRAIN_FILE)
        test_df = pd.read_csv(Config.TEST_FILE)
        
        print(f"Train 데이터 형태: {train_df.shape}")
        print(f"Test 데이터 형태: {test_df.shape}")
        
        # 메모리 최적화
        train_df = optimize_memory_usage(train_df)
        test_df = optimize_memory_usage(test_df)
        
        # 데이터 품질 검사
        train_quality = check_data_quality(train_df, self.feature_columns)
        test_quality = check_data_quality(test_df, self.feature_columns)
        
        if not (train_quality and test_quality):
            print("데이터 품질 문제 발견, 추가 전처리 진행")
            train_df, test_df = self._handle_data_issues(train_df, test_df)
        
        # 기본 통계 저장
        self._calculate_feature_stats(train_df)
        
        return train_df, test_df
    
    def _handle_data_issues(self, train_df, test_df):
        """데이터 이슈 처리"""
        print("데이터 이슈 처리 중...")
        
        # 결측치 처리
        for col in self.feature_columns:
            if train_df[col].isnull().sum() > 0:
                # 중앙값으로 대체
                median_val = train_df[col].median()
                train_df[col].fillna(median_val, inplace=True)
                test_df[col].fillna(median_val, inplace=True)
                print(f"{col} 결측치 {median_val}로 대체")
        
        # 무한값 처리
        for col in self.feature_columns:
            # 무한값을 NaN으로 변환 후 중앙값으로 대체
            train_df[col] = train_df[col].replace([np.inf, -np.inf], np.nan)
            test_df[col] = test_df[col].replace([np.inf, -np.inf], np.nan)
            
            if train_df[col].isnull().sum() > 0:
                median_val = train_df[col].median()
                train_df[col].fillna(median_val, inplace=True)
                test_df[col].fillna(median_val, inplace=True)
                print(f"{col} 무한값 {median_val}로 대체")
        
        return train_df, test_df
    
    def _calculate_feature_stats(self, train_df):
        """피처별 통계 계산"""
        print("피처별 통계 계산 중...")
        
        for col in self.feature_columns:
            stats = {
                'mean': train_df[col].mean(),
                'std': train_df[col].std(),
                'min': train_df[col].min(),
                'max': train_df[col].max(),
                'median': train_df[col].median(),
                'skew': train_df[col].skew(),
                'kurt': train_df[col].kurtosis()
            }
            self.feature_stats[col] = stats
        
        # 고스케일 피처 식별
        self.high_scale_features = []
        self.low_scale_features = []
        
        for col in self.feature_columns:
            max_val = self.feature_stats[col]['max']
            if max_val > 10:
                self.high_scale_features.append(col)
            else:
                self.low_scale_features.append(col)
        
        print(f"고스케일 피처: {len(self.high_scale_features)}개")
        print(f"저스케일 피처: {len(self.low_scale_features)}개")
    
    @timer
    def feature_engineering(self, train_df, test_df):
        """피처 엔지니어링"""
        print("=== 피처 엔지니어링 시작 ===")
        
        X_train = train_df[self.feature_columns].copy()
        X_test = test_df[self.feature_columns].copy()
        y_train = train_df[Config.TARGET_COLUMN]
        
        # 기본 통계 피처 생성
        X_train, X_test = self._create_statistical_features(X_train, X_test)
        
        # 비율 피처 생성
        X_train, X_test = self._create_ratio_features(X_train, X_test)
        
        # 그룹별 피처 생성
        X_train, X_test = self._create_group_features(X_train, X_test)
        
        # 변환 피처 생성
        X_train, X_test = self._create_transform_features(X_train, X_test)
        
        # 전체 데이터에 대한 최종 NaN 및 무한값 처리
        print("최종 데이터 검증 및 정리 중...")
        X_train, X_test = self._final_data_cleanup(X_train, X_test)
        
        return X_train, X_test, y_train
    
    def _final_data_cleanup(self, X_train, X_test):
        """최종 데이터 정리"""
        # 모든 컬럼에 대해 NaN 및 무한값 확인
        print(f"정리 전 - 훈련 데이터 NaN 개수: {X_train.isna().sum().sum()}")
        print(f"정리 전 - 테스트 데이터 NaN 개수: {X_test.isna().sum().sum()}")
        
        # 무한값을 NaN으로 변환
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        
        # NaN 값 처리
        for col in X_train.columns:
            if X_train[col].isna().any():
                # 중앙값으로 대체
                median_val = X_train[col].median()
                if pd.isna(median_val):
                    # 중앙값도 NaN이면 0으로 대체
                    median_val = 0
                
                X_train[col].fillna(median_val, inplace=True)
                X_test[col].fillna(median_val, inplace=True)
                print(f"{col} 컬럼 NaN 값을 {median_val}로 대체")
        
        print(f"정리 후 - 훈련 데이터 NaN 개수: {X_train.isna().sum().sum()}")
        print(f"정리 후 - 테스트 데이터 NaN 개수: {X_test.isna().sum().sum()}")
        
        return X_train, X_test
    
    def _create_statistical_features(self, X_train, X_test):
        """통계적 피처 생성"""
        print("통계적 피처 생성 중...")
        
        # 전체 피처 통계
        X_train['feature_mean'] = X_train[self.feature_columns].mean(axis=1)
        X_train['feature_std'] = X_train[self.feature_columns].std(axis=1)
        X_train['feature_max'] = X_train[self.feature_columns].max(axis=1)
        X_train['feature_min'] = X_train[self.feature_columns].min(axis=1)
        X_train['feature_range'] = X_train['feature_max'] - X_train['feature_min']
        X_train['feature_median'] = X_train[self.feature_columns].median(axis=1)
        
        X_test['feature_mean'] = X_test[self.feature_columns].mean(axis=1)
        X_test['feature_std'] = X_test[self.feature_columns].std(axis=1)
        X_test['feature_max'] = X_test[self.feature_columns].max(axis=1)
        X_test['feature_min'] = X_test[self.feature_columns].min(axis=1)
        X_test['feature_range'] = X_test['feature_max'] - X_test['feature_min']
        X_test['feature_median'] = X_test[self.feature_columns].median(axis=1)
        
        # NaN 값 처리
        stat_features = ['feature_mean', 'feature_std', 'feature_max', 'feature_min', 'feature_range', 'feature_median']
        for feat in stat_features:
            if X_train[feat].isna().any():
                median_val = X_train[feat].median()
                X_train[feat].fillna(median_val, inplace=True)
                X_test[feat].fillna(median_val, inplace=True)
        
        # 분위수 피처
        for q in [0.25, 0.75, 0.9, 0.1]:
            col_name = f'feature_q{int(q*100)}'
            X_train[col_name] = X_train[self.feature_columns].quantile(q, axis=1)
            X_test[col_name] = X_test[self.feature_columns].quantile(q, axis=1)
            
            # NaN 값 처리
            if X_train[col_name].isna().any():
                median_val = X_train[col_name].median()
                X_train[col_name].fillna(median_val, inplace=True)
                X_test[col_name].fillna(median_val, inplace=True)
        
        # 스케일별 통계
        if self.high_scale_features:
            X_train['high_scale_mean'] = X_train[self.high_scale_features].mean(axis=1)
            X_train['high_scale_std'] = X_train[self.high_scale_features].std(axis=1)
            X_test['high_scale_mean'] = X_test[self.high_scale_features].mean(axis=1)
            X_test['high_scale_std'] = X_test[self.high_scale_features].std(axis=1)
            
            # NaN 값 처리
            for feat in ['high_scale_mean', 'high_scale_std']:
                if X_train[feat].isna().any():
                    median_val = X_train[feat].median()
                    X_train[feat].fillna(median_val, inplace=True)
                    X_test[feat].fillna(median_val, inplace=True)
        
        if self.low_scale_features:
            X_train['low_scale_mean'] = X_train[self.low_scale_features].mean(axis=1)
            X_train['low_scale_std'] = X_train[self.low_scale_features].std(axis=1)
            X_test['low_scale_mean'] = X_test[self.low_scale_features].mean(axis=1)
            X_test['low_scale_std'] = X_test[self.low_scale_features].std(axis=1)
            
            # NaN 값 처리
            for feat in ['low_scale_mean', 'low_scale_std']:
                if X_train[feat].isna().any():
                    median_val = X_train[feat].median()
                    X_train[feat].fillna(median_val, inplace=True)
                    X_test[feat].fillna(median_val, inplace=True)
        
        return X_train, X_test
    
    def _create_ratio_features(self, X_train, X_test):
        """비율 피처 생성"""
        print("비율 피처 생성 중...")
        
        # 고스케일 피처들의 비율
        if len(self.high_scale_features) >= 2:
            for i in range(min(3, len(self.high_scale_features))):
                for j in range(i+1, min(3, len(self.high_scale_features))):
                    feat1, feat2 = self.high_scale_features[i], self.high_scale_features[j]
                    ratio_name = f'ratio_{feat1}_{feat2}'
                    
                    # 안전한 나눗셈
                    denominator_train = X_train[feat2] + 1e-8
                    denominator_test = X_test[feat2] + 1e-8
                    
                    X_train[ratio_name] = X_train[feat1] / denominator_train
                    X_test[ratio_name] = X_test[feat1] / denominator_test
                    
                    # NaN 및 무한값 처리
                    X_train[ratio_name] = X_train[ratio_name].replace([np.inf, -np.inf], np.nan)
                    X_test[ratio_name] = X_test[ratio_name].replace([np.inf, -np.inf], np.nan)
                    
                    # NaN을 중앙값으로 대체
                    median_val = X_train[ratio_name].median()
                    X_train[ratio_name].fillna(median_val, inplace=True)
                    X_test[ratio_name].fillna(median_val, inplace=True)
        
        # 통계 기반 비율 (안전한 나눗셈)
        mean_safe = X_train['feature_mean'].abs() + 1e-8
        X_train['max_mean_ratio'] = X_train['feature_max'] / mean_safe
        X_train['range_mean_ratio'] = X_train['feature_range'] / mean_safe
        X_train['std_mean_ratio'] = X_train['feature_std'] / mean_safe
        
        mean_safe_test = X_test['feature_mean'].abs() + 1e-8
        X_test['max_mean_ratio'] = X_test['feature_max'] / mean_safe_test
        X_test['range_mean_ratio'] = X_test['feature_range'] / mean_safe_test
        X_test['std_mean_ratio'] = X_test['feature_std'] / mean_safe_test
        
        # 모든 비율 피처에 대해 NaN 및 무한값 처리
        ratio_features = ['max_mean_ratio', 'range_mean_ratio', 'std_mean_ratio']
        for feat in ratio_features:
            X_train[feat] = X_train[feat].replace([np.inf, -np.inf], np.nan)
            X_test[feat] = X_test[feat].replace([np.inf, -np.inf], np.nan)
            
            median_val = X_train[feat].median()
            X_train[feat].fillna(median_val, inplace=True)
            X_test[feat].fillna(median_val, inplace=True)
        
        return X_train, X_test
    
    def _create_group_features(self, X_train, X_test):
        """그룹별 피처 생성"""
        print("그룹별 피처 생성 중...")
        
        # 10개씩 그룹화
        group_size = 10
        for i in range(0, len(self.feature_columns), group_size):
            end_idx = min(i + group_size, len(self.feature_columns))
            group_features = self.feature_columns[i:end_idx]
            group_name = f'group_{i//group_size + 1}'
            
            X_train[f'{group_name}_mean'] = X_train[group_features].mean(axis=1)
            X_train[f'{group_name}_std'] = X_train[group_features].std(axis=1)
            
            X_test[f'{group_name}_mean'] = X_test[group_features].mean(axis=1)
            X_test[f'{group_name}_std'] = X_test[group_features].std(axis=1)
        
        return X_train, X_test
    
    def _create_transform_features(self, X_train, X_test):
        """변환 피처 생성"""
        print("변환 피처 생성 중...")
        
        # 로그 변환 (양수인 피처만)
        for col in self.feature_columns[:5]:  # 첫 5개만 적용
            if X_train[col].min() > 0:
                X_train[f'{col}_log'] = np.log1p(X_train[col])
                X_test[f'{col}_log'] = np.log1p(X_test[col])
                
                # NaN 및 무한값 처리
                X_train[f'{col}_log'] = X_train[f'{col}_log'].replace([np.inf, -np.inf], np.nan)
                X_test[f'{col}_log'] = X_test[f'{col}_log'].replace([np.inf, -np.inf], np.nan)
                
                # NaN을 중앙값으로 대체
                if X_train[f'{col}_log'].isna().any():
                    median_val = X_train[f'{col}_log'].median()
                    X_train[f'{col}_log'].fillna(median_val, inplace=True)
                    X_test[f'{col}_log'].fillna(median_val, inplace=True)
        
        # 제곱근 변환
        for col in self.feature_columns[:5]:
            if X_train[col].min() >= 0:
                X_train[f'{col}_sqrt'] = np.sqrt(X_train[col])
                X_test[f'{col}_sqrt'] = np.sqrt(X_test[col])
                
                # NaN 및 무한값 처리
                X_train[f'{col}_sqrt'] = X_train[f'{col}_sqrt'].replace([np.inf, -np.inf], np.nan)
                X_test[f'{col}_sqrt'] = X_test[f'{col}_sqrt'].replace([np.inf, -np.inf], np.nan)
                
                # NaN을 중앙값으로 대체
                if X_train[f'{col}_sqrt'].isna().any():
                    median_val = X_train[f'{col}_sqrt'].median()
                    X_train[f'{col}_sqrt'].fillna(median_val, inplace=True)
                    X_test[f'{col}_sqrt'].fillna(median_val, inplace=True)
        
        return X_train, X_test
    
    @timer
    def scale_features(self, X_train, X_test, method='robust'):
        """피처 스케일링"""
        print(f"=== 피처 스케일링 시작 ({method}) ===")
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'quantile':
            self.scaler = QuantileTransformer(random_state=Config.RANDOM_STATE)
        else:
            raise ValueError("지원하지 않는 스케일링 방법입니다.")
        
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
        
        # 스케일링 후 NaN 값 처리
        print(f"스케일링 후 - 훈련 데이터 NaN 개수: {X_train_scaled.isna().sum().sum()}")
        print(f"스케일링 후 - 테스트 데이터 NaN 개수: {X_test_scaled.isna().sum().sum()}")
        
        # NaN 값이 있으면 처리
        if X_train_scaled.isna().sum().sum() > 0 or X_test_scaled.isna().sum().sum() > 0:
            print("스케일링 후 NaN 값 발견, 처리 중...")
            X_train_scaled, X_test_scaled = self._handle_post_scaling_nan(X_train_scaled, X_test_scaled)
        
        # 스케일러 저장
        save_joblib(self.scaler, Config.SCALER_FILE)
        
        return X_train_scaled, X_test_scaled
    
    def _handle_post_scaling_nan(self, X_train, X_test):
        """스케일링 후 NaN 값 처리"""
        from sklearn.impute import SimpleImputer
        
        print("SimpleImputer를 사용한 NaN 값 처리 중...")
        
        # 중앙값 기반 imputer 사용
        imputer = SimpleImputer(strategy='median')
        
        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)
        
        # DataFrame으로 다시 변환
        X_train = pd.DataFrame(
            X_train_imputed,
            columns=X_train.columns,
            index=X_train.index
        )
        X_test = pd.DataFrame(
            X_test_imputed,
            columns=X_test.columns,
            index=X_test.index
        )
        
        print(f"Imputer 적용 후 - 훈련 데이터 NaN 개수: {X_train.isna().sum().sum()}")
        print(f"Imputer 적용 후 - 테스트 데이터 NaN 개수: {X_test.isna().sum().sum()}")
        
        return X_train, X_test
    
    @timer
    def select_features(self, X_train, X_test, y_train, method='mutual_info', k=None):
        """피처 선택"""
        if k is None:
            k = Config.FEATURE_SELECTION_K
        
        print(f"=== 피처 선택 시작 ({method}, k={k}) ===")
        print(f"원본 피처 개수: {X_train.shape[1]}")
        
        # 피처 선택 전 강력한 NaN 값 처리
        print("피처 선택 전 최종 NaN 값 검증...")
        X_train, X_test = self._ensure_no_nan(X_train, X_test)
        
        if method == 'mutual_info':
            self.feature_selector = SelectKBest(mutual_info_classif, k=k)
        elif method == 'f_classif':
            self.feature_selector = SelectKBest(f_classif, k=k)
        elif method == 'chi2':
            # 음수 값이 있으면 최소값을 더해서 양수로 만듦
            if X_train.min().min() < 0:
                offset = abs(X_train.min().min()) + 1
                X_train_pos = X_train + offset
                X_test_pos = X_test + offset
            else:
                X_train_pos = X_train
                X_test_pos = X_test
            
            self.feature_selector = SelectKBest(chi2, k=k)
            X_train_selected = self.feature_selector.fit_transform(X_train_pos, y_train)
            X_test_selected = self.feature_selector.transform(X_test_pos)
            
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
            
            # 피처 선택기 저장
            save_joblib(self.feature_selector, Config.FEATURE_SELECTOR_FILE)
            
            return X_train_selected, X_test_selected
        elif method == 'random_forest':
            return self._select_features_rf(X_train, X_test, y_train, k)
        elif method == 'extra_trees':
            return self._select_features_et(X_train, X_test, y_train, k)
        elif method == 'rfe':
            return self._select_features_rfe(X_train, X_test, y_train, k)
        else:
            raise ValueError("지원하지 않는 피처 선택 방법입니다.")
        
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
        
        # 피처 선택기 저장
        save_joblib(self.feature_selector, Config.FEATURE_SELECTOR_FILE)
        
        return X_train_selected, X_test_selected
    
    def _ensure_no_nan(self, X_train, X_test):
        """NaN 값 완전 제거 보장"""
        from sklearn.impute import SimpleImputer
        
        nan_count_train = X_train.isna().sum().sum()
        nan_count_test = X_test.isna().sum().sum()
        
        print(f"검증 중 - 훈련 데이터 NaN 개수: {nan_count_train}")
        print(f"검증 중 - 테스트 데이터 NaN 개수: {nan_count_test}")
        
        if nan_count_train > 0 or nan_count_test > 0:
            print("NaN 값 발견, SimpleImputer로 강제 처리...")
            
            # 상수 전략으로 0 채우기
            imputer = SimpleImputer(strategy='constant', fill_value=0)
            
            X_train_clean = imputer.fit_transform(X_train)
            X_test_clean = imputer.transform(X_test)
            
            # DataFrame으로 변환
            X_train = pd.DataFrame(
                X_train_clean,
                columns=X_train.columns,
                index=X_train.index
            )
            X_test = pd.DataFrame(
                X_test_clean,
                columns=X_test.columns,
                index=X_test.index
            )
            
            print(f"처리 후 - 훈련 데이터 NaN 개수: {X_train.isna().sum().sum()}")
            print(f"처리 후 - 테스트 데이터 NaN 개수: {X_test.isna().sum().sum()}")
        
        # 무한값도 제거
        X_train = X_train.replace([np.inf, -np.inf], 0)
        X_test = X_test.replace([np.inf, -np.inf], 0)
        
        print("NaN 및 무한값 처리 완료")
        
        return X_train, X_test
    
    def _select_features_rf(self, X_train, X_test, y_train, k):
        """Random Forest 기반 피처 선택"""
        print("Random Forest 기반 피처 중요도 계산 중...")
        
        rf = RandomForestClassifier(
            n_estimators=100, 
            random_state=Config.RANDOM_STATE, 
            n_jobs=Config.N_JOBS
        )
        
        self.feature_selector = SelectFromModel(rf, max_features=k)
        X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
        X_test_selected = self.feature_selector.transform(X_test)
        
        selected_mask = self.feature_selector.get_support()
        self.selected_features = X_train.columns[selected_mask].tolist()
        
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
        
        return X_train_selected, X_test_selected
    
    def _select_features_et(self, X_train, X_test, y_train, k):
        """Extra Trees 기반 피처 선택"""
        print("Extra Trees 기반 피처 중요도 계산 중...")
        
        et = ExtraTreesClassifier(
            n_estimators=100, 
            random_state=Config.RANDOM_STATE, 
            n_jobs=Config.N_JOBS
        )
        
        self.feature_selector = SelectFromModel(et, max_features=k)
        X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
        X_test_selected = self.feature_selector.transform(X_test)
        
        selected_mask = self.feature_selector.get_support()
        self.selected_features = X_train.columns[selected_mask].tolist()
        
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
        
        return X_train_selected, X_test_selected
    
    def _select_features_rfe(self, X_train, X_test, y_train, k):
        """RFE 기반 피처 선택"""
        print("RFE 기반 피처 선택 중...")
        
        estimator = RandomForestClassifier(
            n_estimators=50, 
            random_state=Config.RANDOM_STATE, 
            n_jobs=Config.N_JOBS
        )
        
        self.feature_selector = RFECV(
            estimator, 
            min_features_to_select=k,
            cv=3,
            scoring='f1_macro',
            n_jobs=Config.N_JOBS
        )
        
        X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
        X_test_selected = self.feature_selector.transform(X_test)
        
        selected_mask = self.feature_selector.get_support()
        self.selected_features = X_train.columns[selected_mask].tolist()
        
        print(f"RFE 최적 피처 개수: {self.feature_selector.n_features_}")
        
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
        
        return X_train_selected, X_test_selected
    
    @timer
    def apply_pca(self, X_train, X_test, n_components=None):
        """PCA 적용"""
        if n_components is None:
            n_components = Config.PCA_COMPONENTS
        
        print(f"=== PCA 적용 시작 (n_components={n_components}) ===")
        
        self.pca = PCA(n_components=n_components, random_state=Config.RANDOM_STATE)
        
        X_train_pca = self.pca.fit_transform(X_train)
        X_test_pca = self.pca.transform(X_test)
        
        print(f"PCA 후 컴포넌트 개수: {X_train_pca.shape[1]}")
        print(f"설명 가능한 분산 비율: {self.pca.explained_variance_ratio_.sum():.4f}")
        
        # DataFrame 변환
        pca_columns = [f'PC_{i+1}' for i in range(X_train_pca.shape[1])]
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
        
        # PCA 객체 저장
        save_joblib(self.pca, Config.PCA_FILE)
        
        return X_train_pca, X_test_pca
    
    @timer
    def apply_svd(self, X_train, X_test, n_components=50):
        """SVD 적용"""
        print(f"=== SVD 적용 시작 (n_components={n_components}) ===")
        
        self.svd = TruncatedSVD(
            n_components=n_components, 
            random_state=Config.RANDOM_STATE
        )
        
        X_train_svd = self.svd.fit_transform(X_train)
        X_test_svd = self.svd.transform(X_test)
        
        print(f"SVD 후 컴포넌트 개수: {X_train_svd.shape[1]}")
        print(f"설명 가능한 분산 비율: {self.svd.explained_variance_ratio_.sum():.4f}")
        
        # DataFrame 변환
        svd_columns = [f'SVD_{i+1}' for i in range(X_train_svd.shape[1])]
        X_train_svd = pd.DataFrame(
            X_train_svd, 
            columns=svd_columns, 
            index=X_train.index
        )
        X_test_svd = pd.DataFrame(
            X_test_svd, 
            columns=svd_columns, 
            index=X_test.index
        )
        
        return X_train_svd, X_test_svd
    
    def get_processed_data(self, use_feature_selection=True, use_pca=False, use_svd=False):
        """전체 전처리 파이프라인 실행"""
        print("=== 전체 데이터 전처리 파이프라인 시작 ===")
        
        # 1. 데이터 로드 및 기본 전처리
        train_df, test_df = self.load_and_preprocess_data()
        
        # 2. 피처 엔지니어링
        X_train, X_test, y_train = self.feature_engineering(train_df, test_df)
        
        # 3. 스케일링
        X_train, X_test = self.scale_features(X_train, X_test, method='robust')
        
        # 4. 스케일링 후 추가 안전 체크
        print("스케일링 후 추가 안전 체크...")
        X_train, X_test = self._ensure_no_nan(X_train, X_test)
        
        # 5. 피처 선택
        if use_feature_selection:
            X_train, X_test = self.select_features(
                X_train, X_test, y_train, 
                method='mutual_info', 
                k=Config.FEATURE_SELECTION_K
            )
        
        # 6. PCA
        if use_pca:
            X_train_pca, X_test_pca = self.apply_pca(X_train, X_test)
            X_train = X_train_pca
            X_test = X_test_pca
        
        # 7. SVD
        if use_svd:
            X_train_svd, X_test_svd = self.apply_svd(X_train, X_test)
            X_train = pd.concat([X_train, X_train_svd], axis=1)
            X_test = pd.concat([X_test, X_test_svd], axis=1)
        
        # 8. 최종 검증
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
            print("최종 단계에서 문제 값 발견, 강제 정리...")
            X_train = X_train.replace([np.inf, -np.inf, np.nan], 0)
            X_test = X_test.replace([np.inf, -np.inf, np.nan], 0)
            print("모든 문제 값을 0으로 대체 완료")
        
        print("=== 전처리 완료 ===")
        print(f"최종 피처 개수: {X_train.shape[1]}")
        print(f"최종 데이터 형태 - 훈련: {X_train.shape}, 테스트: {X_test.shape}")
        
        return X_train, X_test, y_train, train_df[Config.ID_COLUMN], test_df[Config.ID_COLUMN]
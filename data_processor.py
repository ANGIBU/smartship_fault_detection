# data_processor.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler, QuantileTransformer, PowerTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif, RFE, SelectFromModel
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
from scipy import stats
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, welch
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
        self.pca = None
        self.power_transformer = None
        self.feature_columns = Config.FEATURE_COLUMNS
        self.selected_features = None
        self.train_stats = {}
        self.sensor_groups = Config.SENSOR_GROUPS
        self.generated_feature_names = []
        
    @timer
    def load_and_preprocess_data(self):
        """데이터 로드 및 전처리"""
        print("=== 데이터 로드 및 전처리 시작 ===")
        
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
            print("데이터 품질 문제 발견, 처리 진행")
            train_df, test_df = self._handle_data_issues(train_df, test_df)
        
        return train_df, test_df
    
    def _handle_data_issues(self, train_df, test_df):
        """데이터 이슈 처리"""
        print("데이터 이슈 처리 중...")
        
        # 무한값 및 결측치 처리
        for col in self.feature_columns:
            # 무한값을 NaN으로 변환
            train_df[col] = train_df[col].replace([np.inf, -np.inf], np.nan)
            test_df[col] = test_df[col].replace([np.inf, -np.inf], np.nan)
            
            # 결측치 처리 - KNN 기반 대체
            if train_df[col].isnull().sum() > 0:
                # 간단한 통계적 대체 먼저 적용
                if train_df[col].std() > 0.01:
                    fill_val = train_df[col].median()
                else:
                    fill_val = train_df[col].mean()
                
                if pd.isna(fill_val):
                    fill_val = 0.0
                    
                train_df[col].fillna(fill_val, inplace=True)
                test_df[col].fillna(fill_val, inplace=True)
        
        return train_df, test_df
    
    def _calculate_sensor_correlations(self, train_df):
        """센서 상관관계 분석"""
        print("센서 상관관계 분석 중...")
        
        # 전체 상관행렬 계산
        corr_matrix = train_df[self.feature_columns].corr().abs()
        
        # 고상관 센서 쌍 식별
        high_corr_pairs = []
        n_features = len(self.feature_columns)
        
        for i in range(n_features):
            for j in range(i+1, n_features):
                corr_val = corr_matrix.iloc[i, j]
                if corr_val > 0.7:
                    high_corr_pairs.append((
                        self.feature_columns[i], 
                        self.feature_columns[j],
                        corr_val
                    ))
        
        # 상위 15개 고상관 쌍 저장
        self.high_corr_pairs = sorted(high_corr_pairs, key=lambda x: x[2], reverse=True)[:15]
        print(f"고상관 센서 쌍: {len(self.high_corr_pairs)}개")
        
        # 센서 그룹별 특성 분석
        self.group_stats = {}
        for group_name, sensors in self.sensor_groups.items():
            valid_sensors = [s for s in sensors if s in self.feature_columns]
            if valid_sensors:
                group_data = train_df[valid_sensors]
                self.group_stats[group_name] = {
                    'sensors': valid_sensors,
                    'mean_corr': group_data.corr().abs().mean().mean(),
                    'variance': group_data.var(),
                    'mean_values': group_data.mean()
                }
        
        # 메모리 정리
        del corr_matrix
        gc.collect()
    
    def _create_statistical_features(self, X_train, X_test):
        """통계 피처 생성"""
        print("통계 피처 생성 중...")
        
        # 기본 통계 피처
        for df in [X_train, X_test]:
            arr = df[self.feature_columns].values
            
            # 1차 통계량
            df['sensor_mean'] = np.mean(arr, axis=1).astype('float32')
            df['sensor_std'] = np.std(arr, axis=1).astype('float32')
            df['sensor_var'] = np.var(arr, axis=1).astype('float32')
            df['sensor_median'] = np.median(arr, axis=1).astype('float32')
            df['sensor_range'] = (np.max(arr, axis=1) - np.min(arr, axis=1)).astype('float32')
            df['sensor_iqr'] = (np.percentile(arr, 75, axis=1) - np.percentile(arr, 25, axis=1)).astype('float32')
            
            # 2차 통계량
            df['sensor_skew'] = skew(arr, axis=1, nan_policy='omit').astype('float32')
            df['sensor_kurtosis'] = kurtosis(arr, axis=1, nan_policy='omit').astype('float32')
            
            # 분위수 기반 피처
            df['sensor_q10'] = np.percentile(arr, 10, axis=1).astype('float32')
            df['sensor_q90'] = np.percentile(arr, 90, axis=1).astype('float32')
            df['sensor_q25'] = np.percentile(arr, 25, axis=1).astype('float32')
            df['sensor_q75'] = np.percentile(arr, 75, axis=1).astype('float32')
            
            # 변동성 피처
            df['sensor_cv'] = (df['sensor_std'] / (df['sensor_mean'] + 1e-8)).astype('float32')
            df['sensor_mad'] = np.mean(np.abs(arr - np.mean(arr, axis=1, keepdims=True)), axis=1).astype('float32')
            
        # 센서 그룹별 통계 피처
        for group_name, stats in self.group_stats.items():
            sensors = stats['sensors']
            if len(sensors) >= 2:
                for df in [X_train, X_test]:
                    group_data = df[sensors].values
                    
                    df[f'{group_name}_mean'] = np.mean(group_data, axis=1).astype('float32')
                    df[f'{group_name}_std'] = np.std(group_data, axis=1).astype('float32')
                    df[f'{group_name}_max'] = np.max(group_data, axis=1).astype('float32')
                    df[f'{group_name}_min'] = np.min(group_data, axis=1).astype('float32')
                    df[f'{group_name}_range'] = (df[f'{group_name}_max'] - df[f'{group_name}_min']).astype('float32')
        
        return X_train, X_test
    
    def _create_interaction_features(self, X_train, X_test):
        """상호작용 피처 생성"""
        print("상호작용 피처 생성 중...")
        
        eps = 1e-8
        
        # 고상관 센서 쌍 기반 피처
        for s1, s2, corr in self.high_corr_pairs[:10]:
            if s1 in X_train.columns and s2 in X_train.columns:
                for df in [X_train, X_test]:
                    # 기본 연산
                    df[f'ratio_{s1}_{s2}'] = (df[s1] / (df[s2].abs() + eps)).astype('float32')
                    df[f'diff_{s1}_{s2}'] = (df[s1] - df[s2]).astype('float32')
                    df[f'sum_{s1}_{s2}'] = (df[s1] + df[s2]).astype('float32')
                    df[f'prod_{s1}_{s2}'] = (df[s1] * df[s2]).astype('float32')
                    
                    # 상관관계 기반 가중합
                    df[f'weighted_{s1}_{s2}'] = (df[s1] * corr + df[s2] * (1 - corr)).astype('float32')
        
        # 센서 그룹 간 상호작용
        group_names = list(self.group_stats.keys())
        for i, group1 in enumerate(group_names):
            for j, group2 in enumerate(group_names[i+1:], i+1):
                if f'{group1}_mean' in X_train.columns and f'{group2}_mean' in X_train.columns:
                    for df in [X_train, X_test]:
                        # 그룹 간 비율
                        df[f'group_ratio_{group1}_{group2}'] = (
                            df[f'{group1}_mean'] / (df[f'{group2}_mean'].abs() + eps)
                        ).astype('float32')
                        
                        # 그룹 간 차분
                        df[f'group_diff_{group1}_{group2}'] = (
                            df[f'{group1}_mean'] - df[f'{group2}_mean']
                        ).astype('float32')
        
        # 다항식 피처 (선택적)
        if Config.POLYNOMIAL_DEGREE > 1:
            important_features = ['sensor_mean', 'sensor_std', 'sensor_range']
            available_features = [f for f in important_features if f in X_train.columns]
            
            for feature in available_features[:3]:  # 상위 3개만
                for df in [X_train, X_test]:
                    df[f'{feature}_squared'] = (df[feature] ** 2).astype('float32')
                    if Config.POLYNOMIAL_DEGREE >= 3:
                        df[f'{feature}_cubed'] = (df[feature] ** 3).astype('float32')
        
        return X_train, X_test
    
    def _create_frequency_features(self, X_train, X_test):
        """주파수 도메인 피처 생성"""
        if not Config.FREQUENCY_FEATURES:
            return X_train, X_test
            
        print("주파수 도메인 피처 생성 중...")
        
        # 각 샘플에 대해 주파수 분석
        for df in [X_train, X_test]:
            fft_features = []
            
            for idx in range(min(len(df), 1000)):  # 처리 속도를 위해 제한
                sample_data = df.iloc[idx][self.feature_columns].values
                
                # FFT 변환
                fft_vals = np.abs(fft(sample_data))
                freqs = fftfreq(len(sample_data))
                
                # 주요 주파수 특성
                dominant_freq = freqs[np.argmax(fft_vals[1:]) + 1]
                spectral_centroid = np.sum(freqs[1:] * fft_vals[1:]) / np.sum(fft_vals[1:])
                spectral_bandwidth = np.sqrt(np.sum(((freqs[1:] - spectral_centroid) ** 2) * fft_vals[1:]) / np.sum(fft_vals[1:]))
                
                fft_features.append([dominant_freq, spectral_centroid, spectral_bandwidth])
            
            # 나머지 샘플은 평균값으로 채움
            if len(fft_features) < len(df):
                avg_features = np.mean(fft_features, axis=0)
                fft_features.extend([avg_features] * (len(df) - len(fft_features)))
            
            fft_array = np.array(fft_features)
            df['dominant_frequency'] = fft_array[:, 0].astype('float32')
            df['spectral_centroid'] = fft_array[:, 1].astype('float32')
            df['spectral_bandwidth'] = fft_array[:, 2].astype('float32')
        
        return X_train, X_test
    
    def _create_rolling_features(self, X_train, X_test):
        """롤링 통계 피처 생성"""
        print("롤링 통계 피처 생성 중...")
        
        # 데이터를 시간 순서로 가정하고 롤링 통계 계산
        for df in [X_train, X_test]:
            for window in Config.ROLLING_WINDOW_SIZES:
                # 중요한 센서들에 대해서만 롤링 통계 적용
                important_sensors = self.feature_columns[:10]  # 처음 10개 센서
                
                for sensor in important_sensors:
                    if sensor in df.columns:
                        # 롤링 평균과 표준편차
                        df[f'{sensor}_rolling_mean_{window}'] = df[sensor].rolling(
                            window=window, min_periods=1
                        ).mean().astype('float32')
                        
                        df[f'{sensor}_rolling_std_{window}'] = df[sensor].rolling(
                            window=window, min_periods=1
                        ).std().fillna(0).astype('float32')
        
        return X_train, X_test
    
    def _create_anomaly_features(self, X_train, X_test):
        """이상 감지 기반 피처 생성"""
        print("이상 감지 피처 생성 중...")
        
        # 각 센서의 정상 범위 정의 (훈련 데이터 기준)
        sensor_ranges = {}
        for sensor in self.feature_columns:
            if sensor in X_train.columns:
                q25 = X_train[sensor].quantile(0.25)
                q75 = X_train[sensor].quantile(0.75)
                iqr = q75 - q25
                
                sensor_ranges[sensor] = {
                    'lower': q25 - 1.5 * iqr,
                    'upper': q75 + 1.5 * iqr,
                    'mean': X_train[sensor].mean(),
                    'std': X_train[sensor].std()
                }
        
        # 이상 감지 피처 생성
        for df in [X_train, X_test]:
            outlier_counts = []
            mahalanobis_distances = []
            z_score_maxes = []
            
            for idx in range(len(df)):
                outlier_count = 0
                z_scores = []
                
                for sensor in self.feature_columns:
                    if sensor in df.columns and sensor in sensor_ranges:
                        value = df.iloc[idx][sensor]
                        ranges = sensor_ranges[sensor]
                        
                        # 아웃라이어 개수
                        if value < ranges['lower'] or value > ranges['upper']:
                            outlier_count += 1
                        
                        # Z-score 계산
                        if ranges['std'] > 0:
                            z_score = abs((value - ranges['mean']) / ranges['std'])
                            z_scores.append(z_score)
                
                outlier_counts.append(outlier_count)
                z_score_maxes.append(max(z_scores) if z_scores else 0)
            
            df['outlier_count'] = pd.Series(outlier_counts, dtype='float32')
            df['max_z_score'] = pd.Series(z_score_maxes, dtype='float32')
            df['outlier_ratio'] = (df['outlier_count'] / len(self.feature_columns)).astype('float32')
        
        return X_train, X_test
    
    @timer
    def feature_engineering(self, train_df, test_df):
        """피처 엔지니어링"""
        print("=== 피처 엔지니어링 시작 ===")
        
        # 메모리 효율적 복사
        X_train = train_df[self.feature_columns].copy()
        X_test = test_df[self.feature_columns].copy()
        y_train = train_df[Config.TARGET_COLUMN].copy()
        
        # 센서 상관관계 분석
        self._calculate_sensor_correlations(train_df)
        
        # 통계 피처 생성
        if Config.STATISTICAL_FEATURES:
            X_train, X_test = self._create_statistical_features(X_train, X_test)
        
        # 상호작용 피처 생성
        if Config.INTERACTION_FEATURES:
            X_train, X_test = self._create_interaction_features(X_train, X_test)
        
        # 주파수 도메인 피처 생성
        if Config.FREQUENCY_FEATURES:
            X_train, X_test = self._create_frequency_features(X_train, X_test)
        
        # 롤링 통계 피처 생성
        X_train, X_test = self._create_rolling_features(X_train, X_test)
        
        # 이상 감지 기반 피처 생성
        X_train, X_test = self._create_anomaly_features(X_train, X_test)
        
        # 최종 데이터 정리
        X_train, X_test = self._final_cleanup(X_train, X_test)
        
        return X_train, X_test, y_train
    
    def _final_cleanup(self, X_train, X_test):
        """최종 데이터 정리"""
        print("최종 데이터 정리 중...")
        
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
        
        print(f"정리 후 훈련 데이터 NaN: {X_train.isna().sum().sum()}")
        print(f"정리 후 테스트 데이터 NaN: {X_test.isna().sum().sum()}")
        print(f"생성된 피처 수: {X_train.shape[1]}")
        
        # 생성된 피처명 저장
        self.generated_feature_names = X_train.columns.tolist()
        
        # 메모리 정리
        gc.collect()
        
        return X_train, X_test
    
    @timer
    def scale_features(self, X_train, X_test, method='quantile'):
        """피처 스케일링"""
        print(f"=== 피처 스케일링 시작 ({method}) ===")
        
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
        elif method == 'power':
            self.power_transformer = PowerTransformer(method='yeo-johnson')
            X_train_scaled = self.power_transformer.fit_transform(X_train)
            X_test_scaled = self.power_transformer.transform(X_test)
            
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
            
            save_joblib(self.power_transformer, Config.SCALER_FILE)
            return X_train_scaled, X_test_scaled
        else:
            self.scaler = QuantileTransformer(output_distribution='normal')
        
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
    def select_features(self, X_train, X_test, y_train, method='hybrid', k=None):
        """피처 선택"""
        if k is None:
            k = Config.FEATURE_SELECTION_K
        
        k = min(k, X_train.shape[1])
        
        print(f"=== 피처 선택 시작 ({method}, k={k}) ===")
        print(f"원본 피처 개수: {X_train.shape[1]}")
        
        if method == 'hybrid':
            # 다중 방법 조합
            selected_features = self._hybrid_feature_selection(X_train, y_train, k)
            
            # 선택된 피처로 데이터 필터링
            X_train_selected = X_train[selected_features]
            X_test_selected = X_test[selected_features]
            
            self.selected_features = selected_features
            
        elif method == 'mutual_info':
            selector = SelectKBest(mutual_info_classif, k=k)
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)
            
            selected_mask = selector.get_support()
            self.selected_features = X_train.columns[selected_mask].tolist()
            
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
            
            self.feature_selector = selector
            save_joblib(self.feature_selector, Config.FEATURE_SELECTOR_FILE)
        
        else:
            # 기본값: mutual_info
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
    
    def _hybrid_feature_selection(self, X_train, y_train, k):
        """하이브리드 피처 선택"""
        print("하이브리드 피처 선택 수행 중...")
        
        feature_scores = {}
        
        # 1. Mutual Information
        try:
            mi_selector = SelectKBest(mutual_info_classif, k=k)
            mi_selector.fit(X_train, y_train)
            mi_scores = mi_selector.scores_
            
            for i, feature in enumerate(X_train.columns):
                feature_scores[feature] = feature_scores.get(feature, 0) + mi_scores[i]
        except:
            print("Mutual Information 계산 실패")
        
        # 2. F-statistic
        try:
            f_selector = SelectKBest(f_classif, k=k)
            f_selector.fit(X_train, y_train)
            f_scores = f_selector.scores_
            
            # 정규화
            f_scores = (f_scores - np.min(f_scores)) / (np.max(f_scores) - np.min(f_scores) + 1e-8)
            
            for i, feature in enumerate(X_train.columns):
                feature_scores[feature] = feature_scores.get(feature, 0) + f_scores[i]
        except:
            print("F-statistic 계산 실패")
        
        # 3. Random Forest 기반
        try:
            rf = RandomForestClassifier(
                n_estimators=100, 
                random_state=Config.RANDOM_STATE,
                n_jobs=2,
                max_depth=5
            )
            rf.fit(X_train, y_train)
            rf_scores = rf.feature_importances_
            
            for i, feature in enumerate(X_train.columns):
                feature_scores[feature] = feature_scores.get(feature, 0) + rf_scores[i]
        except:
            print("Random Forest 중요도 계산 실패")
        
        # 4. Lasso 기반 (L1 정규화)
        try:
            lasso = LassoCV(cv=3, random_state=Config.RANDOM_STATE, max_iter=500)
            lasso.fit(X_train, y_train)
            lasso_scores = np.abs(lasso.coef_)
            
            # 정규화
            if np.max(lasso_scores) > 0:
                lasso_scores = lasso_scores / np.max(lasso_scores)
            
            for i, feature in enumerate(X_train.columns):
                feature_scores[feature] = feature_scores.get(feature, 0) + lasso_scores[i]
        except:
            print("Lasso 계산 실패")
        
        # 점수 기반 상위 k개 피처 선택
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        selected_features = [feature for feature, score in sorted_features[:k]]
        
        print(f"하이브리드 피처 선택 완료: {len(selected_features)}개")
        
        return selected_features
    
    @timer
    def apply_pca(self, X_train, X_test, n_components=0.99):
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
    
    def get_processed_data(self, use_feature_selection=True, use_pca=False, scaling_method='quantile'):
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
                    method='hybrid', 
                    k=available_features
                )
            
            # 5. PCA (선택적)
            if use_pca:
                X_train, X_test = self.apply_pca(X_train, X_test, n_components=Config.PCA_COMPONENTS)
            
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
            
            # ID 컬럼 반환을 위한 원본 로드
            train_ids = pd.read_csv(Config.TRAIN_FILE, usecols=[Config.ID_COLUMN])[Config.ID_COLUMN]
            test_ids = pd.read_csv(Config.TEST_FILE, usecols=[Config.ID_COLUMN])[Config.ID_COLUMN]
            
            return X_train, X_test, y_train, train_ids, test_ids
            
        except Exception as e:
            print(f"데이터 전처리 중 오류 발생: {e}")
            # 메모리 정리
            gc.collect()
            raise
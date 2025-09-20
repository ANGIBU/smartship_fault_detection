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
        # ID 컬럼은 문자열이므로 dtype 지정하지 않음
        
        # 청크 단위로 로드
        train_chunks = []
        test_chunks = []
        
        try:
            # 훈련 데이터 로드
            for chunk in pd.read_csv(Config.TRAIN_FILE, dtype=dtypes, chunksize=Config.CHUNK_SIZE):
                # 메모리 사용량 감소
                if Config.TARGET_COLUMN in chunk.columns:
                    chunk[Config.TARGET_COLUMN] = chunk[Config.TARGET_COLUMN].astype('int16')
                train_chunks.append(chunk)
            
            train_df = pd.concat(train_chunks, ignore_index=True)
            del train_chunks
            gc.collect()
            
            # 테스트 데이터 로드
            for chunk in pd.read_csv(Config.TEST_FILE, dtype=dtypes, chunksize=Config.CHUNK_SIZE):
                test_chunks.append(chunk)
            
            test_df = pd.concat(test_chunks, ignore_index=True)
            del test_chunks
            gc.collect()
            
        except Exception as e:
            print(f"청크 로드 실패, 일반 로드 시도: {e}")
            # 일반 로드 시에도 dtype에서 ID 컬럼 제외
            train_df = pd.read_csv(Config.TRAIN_FILE)
            test_df = pd.read_csv(Config.TEST_FILE)
            
            # 수동으로 데이터 타입 지정
            for col in self.feature_columns:
                if col in train_df.columns:
                    train_df[col] = train_df[col].astype('float32')
                if col in test_df.columns:
                    test_df[col] = test_df[col].astype('float32')
            
            if Config.TARGET_COLUMN in train_df.columns:
                train_df[Config.TARGET_COLUMN] = train_df[Config.TARGET_COLUMN].astype('int16')
        
        print(f"Train 데이터 형태: {train_df.shape}")
        print(f"Test 데이터 형태: {test_df.shape}")
        
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
        
        # 배치 처리로 메모리 절약
        batch_size = 1000
        n_samples_train = len(X_train)
        n_samples_test = len(X_test)
        
        # 결과 저장용 배열
        train_features = {}
        test_features = {}
        
        # 피처 이름 리스트
        feature_names = ['sensor_mean', 'sensor_std', 'sensor_median', 'sensor_min', 'sensor_max', 
                        'sensor_range', 'sensor_q25', 'sensor_q75', 'sensor_iqr', 'sensor_skew', 'sensor_kurtosis']
        
        for name in feature_names:
            train_features[name] = np.zeros(n_samples_train, dtype='float32')
            test_features[name] = np.zeros(n_samples_test, dtype='float32')
        
        # 훈련 데이터 배치 처리
        for i in range(0, n_samples_train, batch_size):
            end_idx = min(i + batch_size, n_samples_train)
            batch_data = X_train[self.feature_columns].iloc[i:end_idx].values
            
            train_features['sensor_mean'][i:end_idx] = np.mean(batch_data, axis=1)
            train_features['sensor_std'][i:end_idx] = np.std(batch_data, axis=1)
            train_features['sensor_median'][i:end_idx] = np.median(batch_data, axis=1)
            train_features['sensor_min'][i:end_idx] = np.min(batch_data, axis=1)
            train_features['sensor_max'][i:end_idx] = np.max(batch_data, axis=1)
            train_features['sensor_range'][i:end_idx] = train_features['sensor_max'][i:end_idx] - train_features['sensor_min'][i:end_idx]
            train_features['sensor_q25'][i:end_idx] = np.percentile(batch_data, 25, axis=1)
            train_features['sensor_q75'][i:end_idx] = np.percentile(batch_data, 75, axis=1)
            train_features['sensor_iqr'][i:end_idx] = train_features['sensor_q75'][i:end_idx] - train_features['sensor_q25'][i:end_idx]
            train_features['sensor_skew'][i:end_idx] = skew(batch_data, axis=1, nan_policy='omit')
            train_features['sensor_kurtosis'][i:end_idx] = kurtosis(batch_data, axis=1, nan_policy='omit')
        
        # 테스트 데이터 배치 처리
        for i in range(0, n_samples_test, batch_size):
            end_idx = min(i + batch_size, n_samples_test)
            batch_data = X_test[self.feature_columns].iloc[i:end_idx].values
            
            test_features['sensor_mean'][i:end_idx] = np.mean(batch_data, axis=1)
            test_features['sensor_std'][i:end_idx] = np.std(batch_data, axis=1)
            test_features['sensor_median'][i:end_idx] = np.median(batch_data, axis=1)
            test_features['sensor_min'][i:end_idx] = np.min(batch_data, axis=1)
            test_features['sensor_max'][i:end_idx] = np.max(batch_data, axis=1)
            test_features['sensor_range'][i:end_idx] = test_features['sensor_max'][i:end_idx] - test_features['sensor_min'][i:end_idx]
            test_features['sensor_q25'][i:end_idx] = np.percentile(batch_data, 25, axis=1)
            test_features['sensor_q75'][i:end_idx] = np.percentile(batch_data, 75, axis=1)
            test_features['sensor_iqr'][i:end_idx] = test_features['sensor_q75'][i:end_idx] - test_features['sensor_q25'][i:end_idx]
            test_features['sensor_skew'][i:end_idx] = skew(batch_data, axis=1, nan_policy='omit')
            test_features['sensor_kurtosis'][i:end_idx] = kurtosis(batch_data, axis=1, nan_policy='omit')
        
        # DataFrame에 추가
        for name in feature_names:
            X_train[name] = train_features[name]
            X_test[name] = test_features[name]
        
        return X_train, X_test
    
    def _create_group_statistics(self, X_train, X_test):
        """센서 그룹별 통계 피처 생성"""
        print("센서 그룹 통계 피처 생성 중")
        
        for group_name, sensors in self.sensor_groups.items():
            valid_sensors = [s for s in sensors if s in self.feature_columns]
            if len(valid_sensors) >= 2:
                # 메모리 효율적 처리
                for df in [X_train, X_test]:
                    group_data = df[valid_sensors].values.astype('float32')
                    
                    df[f'{group_name}_mean'] = np.mean(group_data, axis=1).astype('float32')
                    df[f'{group_name}_std'] = np.std(group_data, axis=1).astype('float32')
                    df[f'{group_name}_max'] = np.max(group_data, axis=1).astype('float32')
                    df[f'{group_name}_min'] = np.min(group_data, axis=1).astype('float32')
                    
                    # 메모리 정리
                    del group_data
        
        return X_train, X_test
    
    @timer
    def feature_engineering(self, train_df, test_df):
        """피처 엔지니어링"""
        print("피처 엔지니어링 시작")
        
        # 메모리 효율적 복사
        X_train = train_df[self.feature_columns].copy()
        X_test = test_df[self.feature_columns].copy()
        y_train = train_df[Config.TARGET_COLUMN].copy()
        
        # 메모리 정리
        del train_df, test_df
        gc.collect()
        
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
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
            
            # NaN을 중앙값으로 대체
            for col in numeric_cols:
                if df[col].isna().any():
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
        
        # 배치 스케일링으로 메모리 절약
        if len(X_train) > 10000:
            # 샘플링하여 스케일러 피팅
            sample_size = min(10000, len(X_train))
            sample_idx = np.random.RandomState(Config.RANDOM_STATE).choice(
                len(X_train), sample_size, replace=False
            )
            X_sample = X_train.iloc[sample_idx]
            self.scaler.fit(X_sample)
            del X_sample
        else:
            self.scaler.fit(X_train)
        
        # 변환 수행
        X_train_scaled = self.scaler.transform(X_train)
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
        
        # 메모리 정리
        gc.collect()
        
        return X_train_scaled, X_test_scaled
    
    @timer
    def select_features(self, X_train, X_test, y_train, method='conservative', k=None):
        """피처 선택 (과적합 방지 강화)"""
        if k is None:
            k = min(25, X_train.shape[1])  # 더 적은 피처 선택
        
        k = min(k, X_train.shape[1])
        
        print(f"피처 선택 시작 ({method}, k={k})")
        print(f"원본 피처 개수: {X_train.shape[1]}")
        
        if method == 'conservative':
            # 여러 방법을 조합한 보수적 피처 선택
            print("보수적 피처 선택 적용")
            
            # 1. 분산 기반 사전 필터링
            from sklearn.feature_selection import VarianceThreshold
            var_threshold = VarianceThreshold(threshold=0.01)
            X_train_var = var_threshold.fit_transform(X_train)
            X_test_var = var_threshold.transform(X_test)
            
            var_features = X_train.columns[var_threshold.get_support()].tolist()
            print(f"분산 필터링 후 피처 수: {len(var_features)}")
            
            # DataFrame 재구성
            X_train_filtered = pd.DataFrame(X_train_var, columns=var_features, index=X_train.index)
            X_test_filtered = pd.DataFrame(X_test_var, columns=var_features, index=X_test.index)
            
            # 2. 상관관계 기반 중복 제거
            corr_matrix = X_train_filtered.corr().abs()
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            # 0.95 이상 상관관계 피처 제거
            high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
            
            remaining_features = [f for f in var_features if f not in high_corr_features]
            print(f"상관관계 필터링 후 피처 수: {len(remaining_features)}")
            
            X_train_corr = X_train_filtered[remaining_features]
            X_test_corr = X_test_filtered[remaining_features]
            
            # 3. 안정적인 피처 선택 방법 조합
            if len(remaining_features) > k:
                # F-test와 mutual_info의 교집합 사용
                from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
                
                # 더 많은 피처를 선택한 후 교집합 구하기
                k_extended = min(k * 2, len(remaining_features))
                
                # F-test 기반 선택
                f_selector = SelectKBest(f_classif, k=k_extended)
                f_selector.fit(X_train_corr, y_train)
                f_features = X_train_corr.columns[f_selector.get_support()].tolist()
                
                # Mutual Info 기반 선택 (샘플링으로 안정성 확보)
                sample_size = min(5000, len(X_train_corr))
                sample_idx = np.random.RandomState(Config.RANDOM_STATE).choice(
                    len(X_train_corr), sample_size, replace=False
                )
                X_sample = X_train_corr.iloc[sample_idx]
                y_sample = y_train.iloc[sample_idx]
                
                mi_selector = SelectKBest(mutual_info_classif, k=k_extended)
                mi_selector.fit(X_sample, y_sample)
                mi_features = X_train_corr.columns[mi_selector.get_support()].tolist()
                
                # 교집합 구하기
                common_features = list(set(f_features) & set(mi_features))
                
                # 교집합이 충분하지 않으면 F-test 결과 우선 사용
                if len(common_features) < k:
                    print(f"교집합 피처 수 부족 ({len(common_features)}), F-test 결과 사용")
                    selected_features = f_features[:k]
                else:
                    # 교집합에서 k개 선택 (F-test 점수 기준)
                    f_scores = f_selector.scores_
                    f_feature_scores = dict(zip(X_train_corr.columns, f_scores))
                    
                    common_scores = [(f, f_feature_scores[f]) for f in common_features]
                    common_scores.sort(key=lambda x: x[1], reverse=True)
                    
                    selected_features = [f for f, _ in common_scores[:k]]
                
                self.selected_features = selected_features
                
            else:
                self.selected_features = remaining_features
            
        elif method == 'mutual_info':
            selector = SelectKBest(mutual_info_classif, k=k)
        elif method == 'f_classif':
            selector = SelectKBest(f_classif, k=k)
        elif method == 'random_forest':
            # 메모리 효율적 RF 피처 선택
            rf = RandomForestClassifier(
                n_estimators=50,  # 메모리 절약
                random_state=Config.RANDOM_STATE,
                n_jobs=1,
                max_depth=5
            )
            
            # 샘플링으로 피팅
            if len(X_train) > 5000:
                sample_idx = np.random.RandomState(Config.RANDOM_STATE).choice(
                    len(X_train), 5000, replace=False
                )
                X_sample = X_train.iloc[sample_idx]
                y_sample = y_train.iloc[sample_idx]
                rf.fit(X_sample, y_sample)
                del X_sample, y_sample
            else:
                rf.fit(X_train, y_train)
            
            importances = rf.feature_importances_
            indices = np.argsort(importances)[::-1][:k]
            
            X_train_selected = X_train.iloc[:, indices].copy()
            X_test_selected = X_test.iloc[:, indices].copy()
            self.selected_features = X_train.columns[indices].tolist()
            
            print(f"선택된 피처 개수: {len(self.selected_features)}")
            
            # 메모리 정리
            del rf
            gc.collect()
            
            return X_train_selected, X_test_selected
        else:
            selector = SelectKBest(mutual_info_classif, k=k)
        
        if method != 'conservative':
            # 샘플링으로 피처 선택
            if len(X_train) > 5000 and method in ['mutual_info', 'f_classif']:
                sample_idx = np.random.RandomState(Config.RANDOM_STATE).choice(
                    len(X_train), 5000, replace=False
                )
                X_sample = X_train.iloc[sample_idx]
                y_sample = y_train.iloc[sample_idx]
                selector.fit(X_sample, y_sample)
                del X_sample, y_sample
            else:
                selector.fit(X_train, y_train)
            
            X_train_selected = selector.transform(X_train)
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
        else:
            # 보수적 방법의 경우
            X_train_selected = X_train[self.selected_features].copy()
            X_test_selected = X_test[self.selected_features].copy()
        
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
            
            # ID 컬럼 저장
            train_ids = train_df[Config.ID_COLUMN].copy()
            test_ids = test_df[Config.ID_COLUMN].copy()
            
            # 2. 피처 엔지니어링
            X_train, X_test, y_train = self.feature_engineering(train_df, test_df)
            
            # 메모리 정리
            del train_df, test_df
            gc.collect()
            
            # 3. 스케일링
            X_train, X_test = self.scale_features(X_train, X_test, scaling_method)
            
            # 4. 피처 선택 (보수적 방법 사용)
            if use_feature_selection:
                available_features = min(25, X_train.shape[1])  # 더 적은 피처 선택
                X_train, X_test = self.select_features(
                    X_train, X_test, y_train, 
                    method='conservative',  # 보수적 피처 선택 사용
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
            
            return X_train, X_test, y_train, train_ids, test_ids
            
        except Exception as e:
            print(f"데이터 전처리 중 오류 발생: {e}")
            # 메모리 정리
            gc.collect()
            raise
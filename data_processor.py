# data_processor.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler, QuantileTransformer, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif, chi2, SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek
from scipy.stats import skew, kurtosis, jarque_bera
from scipy.spatial.distance import pdist, squareform
from scipy.signal import savgol_filter
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
        self.feature_columns = Config.FEATURE_COLUMNS
        self.selected_features = None
        self.feature_importance = {}
        self.interaction_features = []
        self.polynomial_transformer = None
        self.pca_transformer = None
        
    @timer
    def load_data(self):
        """Load data"""
        print("Starting data loading")
        
        try:
            train_df = pd.read_csv(Config.TRAIN_FILE)
            test_df = pd.read_csv(Config.TEST_FILE)
            
            print(f"Train data shape: {train_df.shape}")
            print(f"Test data shape: {test_df.shape}")
            
            # Data type optimization
            for col in self.feature_columns:
                if col in train_df.columns:
                    train_df[col] = train_df[col].astype('float32')
                if col in test_df.columns:
                    test_df[col] = test_df[col].astype('float32')
            
            if Config.TARGET_COLUMN in train_df.columns:
                train_df[Config.TARGET_COLUMN] = train_df[Config.TARGET_COLUMN].astype('int16')
            
            # Data quality check
            train_quality = check_data_quality(train_df, self.feature_columns)
            test_quality = check_data_quality(test_df, self.feature_columns)
            
            if not (train_quality and test_quality):
                print("Handling data quality issues")
                train_df, test_df = self._handle_data_issues(train_df, test_df)
            
            return train_df, test_df
            
        except Exception as e:
            print(f"Data loading failed: {e}")
            raise
    
    @timer
    def load_quick_data(self):
        """Load data for quick mode"""
        print("Loading data for quick mode")
        
        try:
            train_df = pd.read_csv(Config.TRAIN_FILE)
            test_df = pd.read_csv(Config.TEST_FILE)
            
            # Sample data for quick testing
            if len(train_df) > Config.QUICK_SAMPLE_SIZE:
                # Stratified sampling to maintain class distribution
                sample_indices = []
                for class_id in train_df[Config.TARGET_COLUMN].unique():
                    class_data = train_df[train_df[Config.TARGET_COLUMN] == class_id]
                    sample_size = min(len(class_data), Config.QUICK_SAMPLE_SIZE // Config.N_CLASSES)
                    if sample_size > 0:
                        sampled = class_data.sample(n=sample_size, random_state=Config.RANDOM_STATE)
                        sample_indices.extend(sampled.index.tolist())
                
                train_df = train_df.loc[sample_indices].reset_index(drop=True)
                print(f"Sampled training data: {len(train_df)} samples")
            
            # Sample test data
            if len(test_df) > Config.QUICK_SAMPLE_SIZE:
                test_df = test_df.sample(n=Config.QUICK_SAMPLE_SIZE, random_state=Config.RANDOM_STATE).reset_index(drop=True)
                print(f"Sampled test data: {len(test_df)} samples")
            
            # Data type optimization
            for col in self.feature_columns:
                if col in train_df.columns:
                    train_df[col] = train_df[col].astype('float32')
                if col in test_df.columns:
                    test_df[col] = test_df[col].astype('float32')
            
            if Config.TARGET_COLUMN in train_df.columns:
                train_df[Config.TARGET_COLUMN] = train_df[Config.TARGET_COLUMN].astype('int16')
            
            # Basic data cleaning
            train_df, test_df = self._handle_data_issues(train_df, test_df)
            
            return train_df, test_df
            
        except Exception as e:
            print(f"Quick data loading failed: {e}")
            raise
    
    def _handle_data_issues(self, train_df, test_df):
        """Handle data issues"""
        for col in self.feature_columns:
            if col in train_df.columns and col in test_df.columns:
                # Handle infinite values
                train_df[col] = train_df[col].replace([np.inf, -np.inf], np.nan)
                test_df[col] = test_df[col].replace([np.inf, -np.inf], np.nan)
                
                # Handle missing values
                if train_df[col].isnull().sum() > 0 or test_df[col].isnull().sum() > 0:
                    fill_val = train_df[col].median()
                    if pd.isna(fill_val):
                        fill_val = 0.0
                    
                    train_df[col].fillna(fill_val, inplace=True)
                    test_df[col].fillna(fill_val, inplace=True)
        
        return train_df, test_df
    
    @timer
    def create_statistical_features(self, X_train, X_test):
        """Create statistical features"""
        print("Creating statistical features")
        
        for df in [X_train, X_test]:
            sensor_data = df[self.feature_columns].values
            
            # Basic statistics
            if 'mean' in Config.STATISTICAL_FEATURES:
                df['sensor_mean'] = np.mean(sensor_data, axis=1).astype('float32')
            
            if 'std' in Config.STATISTICAL_FEATURES:
                df['sensor_std'] = np.std(sensor_data, axis=1).astype('float32')
            
            if 'median' in Config.STATISTICAL_FEATURES:
                df['sensor_median'] = np.median(sensor_data, axis=1).astype('float32')
            
            if 'min' in Config.STATISTICAL_FEATURES:
                df['sensor_min'] = np.min(sensor_data, axis=1).astype('float32')
            
            if 'max' in Config.STATISTICAL_FEATURES:
                df['sensor_max'] = np.max(sensor_data, axis=1).astype('float32')
            
            if 'range' in Config.STATISTICAL_FEATURES:
                df['sensor_range'] = (df.get('sensor_max', np.max(sensor_data, axis=1)) - 
                                     df.get('sensor_min', np.min(sensor_data, axis=1))).astype('float32')
            
            # Distribution characteristics
            if 'skew' in Config.STATISTICAL_FEATURES:
                df['sensor_skew'] = skew(sensor_data, axis=1, nan_policy='omit').astype('float32')
            
            if 'kurtosis' in Config.STATISTICAL_FEATURES:
                df['sensor_kurtosis'] = kurtosis(sensor_data, axis=1, nan_policy='omit').astype('float32')
            
            # Quantiles
            if 'q25' in Config.STATISTICAL_FEATURES:
                df['sensor_q25'] = np.percentile(sensor_data, 25, axis=1).astype('float32')
            
            if 'q75' in Config.STATISTICAL_FEATURES:
                df['sensor_q75'] = np.percentile(sensor_data, 75, axis=1).astype('float32')
            
            if 'iqr' in Config.STATISTICAL_FEATURES:
                q75 = df.get('sensor_q75', np.percentile(sensor_data, 75, axis=1))
                q25 = df.get('sensor_q25', np.percentile(sensor_data, 25, axis=1))
                df['sensor_iqr'] = (q75 - q25).astype('float32')
            
            # Variability indicators
            if 'cv' in Config.STATISTICAL_FEATURES:
                mean_vals = df.get('sensor_mean', np.mean(sensor_data, axis=1))
                std_vals = df.get('sensor_std', np.std(sensor_data, axis=1))
                df['sensor_cv'] = (std_vals / (np.abs(mean_vals) + 1e-8)).astype('float32')
            
            # Count-based features
            if 'zero_count' in Config.STATISTICAL_FEATURES:
                df['sensor_zero_count'] = (sensor_data == 0).sum(axis=1).astype('float32')
            
            if 'negative_count' in Config.STATISTICAL_FEATURES:
                df['sensor_negative_count'] = (sensor_data < 0).sum(axis=1).astype('float32')
            
            if 'positive_count' in Config.STATISTICAL_FEATURES:
                df['sensor_positive_count'] = (sensor_data > 0).sum(axis=1).astype('float32')
            
            # Outlier detection
            if 'outlier_count' in Config.STATISTICAL_FEATURES:
                q75 = df.get('sensor_q75', np.percentile(sensor_data, 75, axis=1)).values.reshape(-1, 1)
                q25 = df.get('sensor_q25', np.percentile(sensor_data, 25, axis=1)).values.reshape(-1, 1)
                iqr = (q75 - q25)
                
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                
                outliers = ((sensor_data < lower_bound) | (sensor_data > upper_bound)).sum(axis=1)
                df['sensor_outlier_count'] = outliers.astype('float32')
                
                if 'outlier_ratio' in Config.STATISTICAL_FEATURES:
                    df['sensor_outlier_ratio'] = (outliers / len(self.feature_columns)).astype('float32')
            
            # Time series features
            df['sensor_rms'] = np.sqrt(np.mean(sensor_data**2, axis=1)).astype('float32')
            df['sensor_energy'] = np.sum(sensor_data**2, axis=1).astype('float32')
            df['sensor_peak_to_peak'] = (np.max(sensor_data, axis=1) - np.min(sensor_data, axis=1)).astype('float32')
            
            # Percentile features
            for percentile in [10, 90, 95]:
                col_name = f'sensor_p{percentile}'
                df[col_name] = np.percentile(sensor_data, percentile, axis=1).astype('float32')
            
            # Moment features
            df['sensor_variance'] = np.var(sensor_data, axis=1).astype('float32')
            df['sensor_mad'] = np.median(np.abs(sensor_data - np.median(sensor_data, axis=1, keepdims=True)), axis=1).astype('float32')
            
            # Entropy-like features
            df['sensor_entropy'] = self._calculate_entropy(sensor_data).astype('float32')
        
        return X_train, X_test
    
    def _calculate_entropy(self, data):
        """Calculate entropy-like measure for sensor data"""
        entropies = []
        for row in data:
            # Discretize data into bins
            hist, _ = np.histogram(row, bins=10, density=True)
            hist = hist + 1e-10  # Avoid log(0)
            entropy = -np.sum(hist * np.log(hist))
            entropies.append(entropy)
        return np.array(entropies)
    
    @timer
    def create_domain_features(self, X_train, X_test):
        """Create domain-specific features for equipment monitoring"""
        print("Creating domain-specific features")
        
        for df in [X_train, X_test]:
            sensor_data = df[self.feature_columns].values
            
            # Equipment stability indicators
            df['stability_index'] = (1 / (1 + df['sensor_cv'])).astype('float32')
            df['consistency_score'] = (1 - df['sensor_range'] / (df['sensor_max'] + 1e-8)).astype('float32')
            
            # Fault detection indicators
            df['anomaly_strength'] = (df['sensor_outlier_ratio'] * df['sensor_cv']).astype('float32')
            df['deviation_magnitude'] = np.sqrt(df['sensor_variance']).astype('float32')
            
            # Signal quality measures
            df['signal_clarity'] = (df['sensor_energy'] / (df['sensor_std'] + 1e-8)).astype('float32')
            df['noise_level'] = (df['sensor_mad'] / (df['sensor_median'] + 1e-8)).astype('float32')
            
            # Operational state features
            df['operational_efficiency'] = (df['sensor_mean'] / (df['sensor_max'] + 1e-8)).astype('float32')
            df['load_factor'] = (df['sensor_p90'] / (df['sensor_p10'] + 1e-8)).astype('float32')
            
            # Cross-sensor relationships
            df['sensor_balance'] = (df['sensor_std'] / (df['sensor_mean'] + 1e-8)).astype('float32')
            df['uniformity_score'] = (1 / (1 + df['sensor_skew']**2 + df['sensor_kurtosis']**2)).astype('float32')
        
        return X_train, X_test
    
    @timer
    def create_quick_statistical_features(self, X_train, X_test):
        """Create minimal statistical features for quick mode"""
        print("Creating basic statistical features for quick mode")
        
        for df in [X_train, X_test]:
            sensor_data = df[self.feature_columns].values
            
            # Only essential statistics
            df['sensor_mean'] = np.mean(sensor_data, axis=1).astype('float32')
            df['sensor_std'] = np.std(sensor_data, axis=1).astype('float32')
            df['sensor_range'] = (np.max(sensor_data, axis=1) - np.min(sensor_data, axis=1)).astype('float32')
        
        return X_train, X_test
    
    @timer
    def create_correlation_features(self, X_train, X_test):
        """Create correlation-based features"""
        print("Creating correlation features")
        
        # Calculate correlation from training data
        train_corr = X_train[self.feature_columns].corr()
        
        # Identify highly correlated sensor pairs
        high_corr_pairs = []
        for i in range(len(self.feature_columns)):
            for j in range(i+1, len(self.feature_columns)):
                corr_val = train_corr.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr_pairs.append((self.feature_columns[i], self.feature_columns[j], corr_val))
        
        print(f"High correlation sensor pairs: {len(high_corr_pairs)}")
        
        # Create correlation-based features
        for df in [X_train, X_test]:
            for sensor1, sensor2, corr_val in high_corr_pairs[:20]:
                if sensor1 in df.columns and sensor2 in df.columns:
                    # Ratio
                    df[f'{sensor1}_{sensor2}_ratio'] = (df[sensor1] / (df[sensor2] + 1e-8)).astype('float32')
                    # Difference
                    df[f'{sensor1}_{sensor2}_diff'] = (df[sensor1] - df[sensor2]).astype('float32')
                    # Product
                    df[f'{sensor1}_{sensor2}_product'] = (df[sensor1] * df[sensor2]).astype('float32')
                    
                    # Correlation strength features
                    if abs(corr_val) > 0.85:
                        df[f'{sensor1}_{sensor2}_norm_diff'] = (
                            np.abs(df[sensor1] - df[sensor2]) / (np.abs(df[sensor1]) + np.abs(df[sensor2]) + 1e-8)
                        ).astype('float32')
        
        return X_train, X_test
    
    @timer
    def create_interaction_features(self, X_train, X_test):
        """Create interaction features between top sensors"""
        if not Config.CREATE_INTERACTION_FEATURES:
            return X_train, X_test
            
        print("Creating interaction features")
        
        # Identify top important sensors
        top_sensors = ['X_42', 'X_19', 'X_40', 'X_46', 'X_35', 'X_23', 'X_17', 'X_29', 'X_31', 'X_08']
        available_sensors = [s for s in top_sensors if s in X_train.columns][:Config.INTERACTION_TOP_N]
        
        for df in [X_train, X_test]:
            # Pairwise interactions
            for i in range(len(available_sensors)):
                for j in range(i+1, len(available_sensors)):
                    sensor1, sensor2 = available_sensors[i], available_sensors[j]
                    
                    # Multiplicative interaction
                    df[f'{sensor1}_x_{sensor2}'] = (df[sensor1] * df[sensor2]).astype('float32')
                    
                    # Additive interaction
                    df[f'{sensor1}_plus_{sensor2}'] = (df[sensor1] + df[sensor2]).astype('float32')
                    
                    # Minimum and maximum
                    df[f'{sensor1}_{sensor2}_min'] = np.minimum(df[sensor1], df[sensor2]).astype('float32')
                    df[f'{sensor1}_{sensor2}_max'] = np.maximum(df[sensor1], df[sensor2]).astype('float32')
                    
                    self.interaction_features.extend([
                        f'{sensor1}_x_{sensor2}', f'{sensor1}_plus_{sensor2}',
                        f'{sensor1}_{sensor2}_min', f'{sensor1}_{sensor2}_max'
                    ])
            
            # Three-way interactions for top 5 sensors
            top_3_sensors = available_sensors[:5]
            for i in range(len(top_3_sensors)):
                for j in range(i+1, len(top_3_sensors)):
                    for k in range(j+1, len(top_3_sensors)):
                        s1, s2, s3 = top_3_sensors[i], top_3_sensors[j], top_3_sensors[k]
                        df[f'{s1}_{s2}_{s3}_mean'] = ((df[s1] + df[s2] + df[s3]) / 3).astype('float32')
                        df[f'{s1}_{s2}_{s3}_product'] = (df[s1] * df[s2] * df[s3]).astype('float32')
                        
                        self.interaction_features.extend([
                            f'{s1}_{s2}_{s3}_mean', f'{s1}_{s2}_{s3}_product'
                        ])
        
        print(f"Created {len(self.interaction_features)} interaction features")
        return X_train, X_test
    
    @timer
    def create_polynomial_features(self, X_train, X_test):
        """Create polynomial features for top sensors"""
        if not Config.CREATE_POLYNOMIAL_FEATURES:
            return X_train, X_test
            
        print("Creating polynomial features")
        
        # Select top sensors for polynomial features
        top_sensors = ['X_42', 'X_19', 'X_40', 'X_46', 'X_35', 'X_23']
        available_sensors = [s for s in top_sensors if s in X_train.columns]
        
        if len(available_sensors) == 0:
            return X_train, X_test
        
        # Create polynomial features for selected sensors
        for df in [X_train, X_test]:
            for sensor in available_sensors:
                # Power transformations
                df[f'{sensor}_squared'] = (df[sensor] ** 2).astype('float32')
                df[f'{sensor}_cubed'] = (df[sensor] ** 3).astype('float32')
                
                # Root transformations
                df[f'{sensor}_sqrt'] = np.sign(df[sensor]) * np.sqrt(np.abs(df[sensor])).astype('float32')
                df[f'{sensor}_cbrt'] = np.sign(df[sensor]) * np.power(np.abs(df[sensor]), 1/3).astype('float32')
                
                # Log transformations
                df[f'{sensor}_log'] = np.sign(df[sensor]) * np.log1p(np.abs(df[sensor])).astype('float32')
                df[f'{sensor}_log10'] = np.sign(df[sensor]) * np.log10(np.abs(df[sensor]) + 1).astype('float32')
                
                # Exponential transformations (with clipping to avoid overflow)
                clipped_sensor = np.clip(df[sensor], -10, 10)
                df[f'{sensor}_exp'] = np.exp(clipped_sensor).astype('float32')
                df[f'{sensor}_tanh'] = np.tanh(df[sensor]).astype('float32')
                
                # Reciprocal transformation
                df[f'{sensor}_reciprocal'] = (1 / (df[sensor] + 1e-8)).astype('float32')
        
        return X_train, X_test
    
    @timer
    def create_pca_features(self, X_train, X_test, n_components=10):
        """Create PCA features from sensor data"""
        print(f"Creating PCA features with {n_components} components")
        
        try:
            # Use only original sensor features for PCA
            sensor_cols = [col for col in self.feature_columns if col in X_train.columns]
            
            if len(sensor_cols) < n_components:
                n_components = len(sensor_cols)
            
            # Fit PCA on training data
            self.pca_transformer = PCA(n_components=n_components, random_state=Config.RANDOM_STATE)
            train_pca = self.pca_transformer.fit_transform(X_train[sensor_cols])
            test_pca = self.pca_transformer.transform(X_test[sensor_cols])
            
            # Add PCA features
            for i in range(n_components):
                X_train[f'pca_{i+1:02d}'] = train_pca[:, i].astype('float32')
                X_test[f'pca_{i+1:02d}'] = test_pca[:, i].astype('float32')
            
            # Calculate explained variance ratio
            explained_variance = self.pca_transformer.explained_variance_ratio_
            total_explained = np.sum(explained_variance)
            
            print(f"PCA explained variance: {total_explained:.3f}")
            
        except Exception as e:
            print(f"PCA feature creation failed: {e}")
        
        return X_train, X_test
    
    @timer
    def create_time_series_features(self, X_train, X_test):
        """Create time series features assuming sensor order represents time"""
        print("Creating time series features")
        
        for df in [X_train, X_test]:
            sensor_data = df[self.feature_columns].values
            
            # Trend features
            trends = []
            for row in sensor_data:
                # Calculate linear trend
                x = np.arange(len(row))
                trend = np.polyfit(x, row, 1)[0]  # Slope of linear fit
                trends.append(trend)
            
            df['sensor_trend'] = np.array(trends).astype('float32')
            
            # Smoothing features
            try:
                smoothed_data = []
                for row in sensor_data:
                    if len(row) >= 5:  # Need at least 5 points for smoothing
                        smoothed = savgol_filter(row, window_length=5, polyorder=2)
                        smoothed_data.append(np.mean(smoothed))
                    else:
                        smoothed_data.append(np.mean(row))
                
                df['sensor_smoothed_mean'] = np.array(smoothed_data).astype('float32')
                
                # Calculate difference between original and smoothed
                original_means = np.mean(sensor_data, axis=1)
                df['sensor_smoothness'] = (original_means - df['sensor_smoothed_mean']).astype('float32')
                
            except Exception as e:
                print(f"Smoothing features failed: {e}")
                df['sensor_smoothed_mean'] = np.mean(sensor_data, axis=1).astype('float32')
                df['sensor_smoothness'] = np.zeros(len(df)).astype('float32')
            
            # Seasonality indicators (assuming some periodic patterns)
            df['sensor_first_half_mean'] = np.mean(sensor_data[:, :len(self.feature_columns)//2], axis=1).astype('float32')
            df['sensor_second_half_mean'] = np.mean(sensor_data[:, len(self.feature_columns)//2:], axis=1).astype('float32')
            df['sensor_half_diff'] = (df['sensor_first_half_mean'] - df['sensor_second_half_mean']).astype('float32')
        
        return X_train, X_test
    
    @timer
    def handle_class_imbalance(self, X_train, y_train, method='smote'):
        """Handle class imbalance with multiple techniques"""
        print(f"Starting class imbalance handling with {method}")
        
        # Check class distribution
        class_counts = y_train.value_counts().sort_index()
        print("Class sample counts:")
        for class_id, count in class_counts.items():
            if class_id < 10:
                print(f"  Class {class_id}: {count} samples")
        
        # Calculate imbalance degree
        max_count = class_counts.max()
        min_count = class_counts.min()
        imbalance_ratio = max_count / min_count
        
        print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 2.5:
            print(f"Applying {method.upper()}")
            
            try:
                # Calculate appropriate k_neighbors
                min_class_samples = class_counts.min()
                k_neighbors = min(5, min_class_samples - 1)
                
                if k_neighbors < 1:
                    print("Resampling not applicable, using original data")
                    return X_train, y_train
                
                # Apply different resampling strategies
                if method == 'smote':
                    resampler = SMOTE(
                        sampling_strategy='auto',
                        random_state=Config.RANDOM_STATE,
                        k_neighbors=k_neighbors
                    )
                elif method == 'adasyn':
                    resampler = ADASYN(
                        sampling_strategy='auto',
                        random_state=Config.RANDOM_STATE,
                        n_neighbors=k_neighbors
                    )
                elif method == 'smoteenn':
                    resampler = SMOTEENN(
                        sampling_strategy='auto',
                        random_state=Config.RANDOM_STATE,
                        smote=SMOTE(random_state=Config.RANDOM_STATE, k_neighbors=k_neighbors)
                    )
                elif method == 'smotetomek':
                    resampler = SMOTETomek(
                        sampling_strategy='auto',
                        random_state=Config.RANDOM_STATE,
                        smote=SMOTE(random_state=Config.RANDOM_STATE, k_neighbors=k_neighbors)
                    )
                elif method == 'balanced':
                    # Custom balanced sampling
                    return self._balanced_sampling(X_train, y_train)
                else:
                    print(f"Unknown method {method}, using SMOTE")
                    resampler = SMOTE(
                        sampling_strategy='auto',
                        random_state=Config.RANDOM_STATE,
                        k_neighbors=k_neighbors
                    )
                
                X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)
                
                print(f"Before resampling: {X_train.shape}")
                print(f"After resampling: {X_resampled.shape}")
                
                return X_resampled, y_resampled
                
            except Exception as e:
                print(f"{method.upper()} application failed: {e}")
                return X_train, y_train
        else:
            print("Class balance is appropriate, skipping resampling")
            return X_train, y_train
    
    def _balanced_sampling(self, X_train, y_train):
        """Custom balanced sampling method"""
        class_counts = y_train.value_counts()
        target_count = int(class_counts.median())
        
        balanced_indices = []
        
        for class_id in class_counts.index:
            class_indices = y_train[y_train == class_id].index.tolist()
            
            if len(class_indices) >= target_count:
                # Undersample
                sampled_indices = np.random.choice(class_indices, target_count, replace=False)
            else:
                # Oversample
                sampled_indices = np.random.choice(class_indices, target_count, replace=True)
            
            balanced_indices.extend(sampled_indices)
        
        X_balanced = X_train.iloc[balanced_indices].reset_index(drop=True)
        y_balanced = y_train.iloc[balanced_indices].reset_index(drop=True)
        
        return X_balanced, y_balanced
    
    @timer
    def scale_features(self, X_train, X_test, method='robust'):
        """Feature scaling with multiple methods"""
        print(f"Starting feature scaling ({method})")
        
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
        
        # Scale only numeric columns
        numeric_columns = X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        # Apply scaling
        X_train_scaled[numeric_columns] = self.scaler.fit_transform(X_train[numeric_columns])
        X_test_scaled[numeric_columns] = self.scaler.transform(X_test[numeric_columns])
        
        # Maintain data types
        for col in numeric_columns:
            X_train_scaled[col] = X_train_scaled[col].astype('float32')
            X_test_scaled[col] = X_test_scaled[col].astype('float32')
        
        save_joblib(self.scaler, Config.SCALER_FILE)
        
        return X_train_scaled, X_test_scaled
    
    @timer
    def select_features(self, X_train, X_test, y_train, n_features=None):
        """Feature selection with multiple methods"""
        if n_features is None:
            n_features = Config.TARGET_FEATURES
        
        n_features = min(n_features, X_train.shape[1])
        
        print(f"Starting feature selection (target: {n_features} features)")
        print(f"Original feature count: {X_train.shape[1]}")
        
        # Remove zero variance features
        constant_features = []
        for col in X_train.columns:
            if X_train[col].nunique() <= 1:
                constant_features.append(col)
        
        if constant_features:
            print(f"Removing constant features: {len(constant_features)}")
            X_train = X_train.drop(columns=constant_features)
            X_test = X_test.drop(columns=constant_features)
        
        # Apply multiple feature selection methods with different weights
        feature_scores = {}
        
        # Mutual Information (weight: 0.35)
        try:
            mi_selector = SelectKBest(mutual_info_classif, k=n_features)
            mi_selector.fit(X_train, y_train)
            mi_scores = mi_selector.scores_
            
            for i, col in enumerate(X_train.columns):
                if col not in feature_scores:
                    feature_scores[col] = 0
                feature_scores[col] += 0.35 * mi_scores[i] / np.max(mi_scores)
        except Exception as e:
            print(f"Mutual Information calculation failed: {e}")
        
        # F-test (weight: 0.25)
        try:
            f_selector = SelectKBest(f_classif, k=n_features)
            f_selector.fit(X_train, y_train)
            f_scores = f_selector.scores_
            
            for i, col in enumerate(X_train.columns):
                if col not in feature_scores:
                    feature_scores[col] = 0
                feature_scores[col] += 0.25 * f_scores[i] / np.max(f_scores)
        except Exception as e:
            print(f"F-test calculation failed: {e}")
        
        # Tree-based feature importance (weight: 0.25)
        try:
            rf_selector = RandomForestClassifier(
                n_estimators=100, 
                random_state=Config.RANDOM_STATE,
                n_jobs=2,
                max_depth=8
            )
            rf_selector.fit(X_train, y_train)
            rf_scores = rf_selector.feature_importances_
            
            for i, col in enumerate(X_train.columns):
                if col not in feature_scores:
                    feature_scores[col] = 0
                feature_scores[col] += 0.25 * rf_scores[i]
        except Exception as e:
            print(f"Random Forest feature importance calculation failed: {e}")
        
        # Variance-based selection (weight: 0.15)
        try:
            variances = X_train.var()
            normalized_variances = variances / variances.max()
            
            for col in X_train.columns:
                if col not in feature_scores:
                    feature_scores[col] = 0
                feature_scores[col] += 0.15 * normalized_variances[col]
        except Exception as e:
            print(f"Variance calculation failed: {e}")
        
        # Score-based feature selection
        if feature_scores:
            sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
            self.selected_features = [feat for feat, score in sorted_features[:n_features]]
            
            # Save feature importance
            self.feature_importance = dict(sorted_features[:n_features])
        else:
            # Use original sensor features by default
            available_sensors = [col for col in self.feature_columns if col in X_train.columns]
            self.selected_features = available_sensors[:n_features]
            print("Feature selection failed, using original sensor features")
        
        X_train_selected = X_train[self.selected_features].copy()
        X_test_selected = X_test[self.selected_features].copy()
        
        print(f"Selected feature count: {len(self.selected_features)}")
        
        # Analyze selected feature types
        original_count = sum(1 for f in self.selected_features if f in self.feature_columns)
        statistical_count = sum(1 for f in self.selected_features if 'sensor_' in f)
        interaction_count = sum(1 for f in self.selected_features if ('_x_' in f or '_plus_' in f or '_min' in f or '_max' in f))
        correlation_count = sum(1 for f in self.selected_features if ('_ratio' in f or '_diff' in f or '_product' in f))
        polynomial_count = sum(1 for f in self.selected_features if ('_squared' in f or '_sqrt' in f or '_log' in f or '_cubed' in f))
        pca_count = sum(1 for f in self.selected_features if f.startswith('pca_'))
        domain_count = sum(1 for f in self.selected_features if f in ['stability_index', 'consistency_score', 'anomaly_strength', 'deviation_magnitude'])
        other_count = len(self.selected_features) - original_count - statistical_count - interaction_count - correlation_count - polynomial_count - pca_count - domain_count
        
        print(f"Feature type distribution:")
        print(f"  Original sensors: {original_count}")
        print(f"  Statistical: {statistical_count}")
        print(f"  Interaction: {interaction_count}")
        print(f"  Correlation: {correlation_count}")
        print(f"  Polynomial: {polynomial_count}")
        print(f"  PCA: {pca_count}")
        print(f"  Domain: {domain_count}")
        if other_count > 0:
            print(f"  Other: {other_count}")
        
        return X_train_selected, X_test_selected
    
    @timer
    def quick_feature_selection(self, X_train, X_test, y_train):
        """Quick feature selection for quick mode"""
        print(f"Quick feature selection (target: {Config.QUICK_FEATURE_COUNT} features)")
        
        # Remove constant features
        constant_features = []
        for col in X_train.columns:
            if X_train[col].nunique() <= 1:
                constant_features.append(col)
        
        if constant_features:
            X_train = X_train.drop(columns=constant_features)
            X_test = X_test.drop(columns=constant_features)
        
        # Simple variance-based selection
        variances = X_train.var()
        top_variance_features = variances.nlargest(Config.QUICK_FEATURE_COUNT).index.tolist()
        
        self.selected_features = top_variance_features
        
        X_train_selected = X_train[self.selected_features].copy()
        X_test_selected = X_test[self.selected_features].copy()
        
        print(f"Selected {len(self.selected_features)} features by variance")
        
        return X_train_selected, X_test_selected
    
    @timer
    def get_processed_data(self, use_resampling=True, scaling_method='robust', resampling_method='smote'):
        """Execute complete data preprocessing pipeline"""
        print("Starting complete data preprocessing pipeline")
        
        try:
            # 1. Data loading
            train_df, test_df = self.load_data()
            
            # Save ID columns
            train_ids = train_df[Config.ID_COLUMN].copy()
            test_ids = test_df[Config.ID_COLUMN].copy()
            
            # Separate features and target
            X_train = train_df[self.feature_columns].copy()
            X_test = test_df[self.feature_columns].copy()
            y_train = train_df[Config.TARGET_COLUMN].copy()
            
            del train_df, test_df
            gc.collect()
            
            # 2. Create statistical features
            X_train, X_test = self.create_statistical_features(X_train, X_test)
            
            # 3. Create domain-specific features
            X_train, X_test = self.create_domain_features(X_train, X_test)
            
            # 4. Create correlation features
            X_train, X_test = self.create_correlation_features(X_train, X_test)
            
            # 5. Create interaction features
            X_train, X_test = self.create_interaction_features(X_train, X_test)
            
            # 6. Create polynomial features
            X_train, X_test = self.create_polynomial_features(X_train, X_test)
            
            # 7. Create PCA features
            X_train, X_test = self.create_pca_features(X_train, X_test, n_components=8)
            
            # 8. Create time series features
            X_train, X_test = self.create_time_series_features(X_train, X_test)
            
            # 9. Scaling
            X_train, X_test = self.scale_features(X_train, X_test, scaling_method)
            
            # 10. Feature selection
            X_train, X_test = self.select_features(X_train, X_test, y_train)
            
            # 11. Handle class imbalance
            if use_resampling:
                X_train, y_train = self.handle_class_imbalance(X_train, y_train, resampling_method)
                # Adjust IDs after resampling
                if len(X_train) != len(train_ids):
                    train_ids = pd.Series(range(len(X_train)), name=Config.ID_COLUMN)
            
            # 12. Final validation
            print("Final data validation")
            
            train_nan_count = X_train.isna().sum().sum()
            test_nan_count = X_test.isna().sum().sum()
            
            print(f"Final training data NaN: {train_nan_count}")
            print(f"Final test data NaN: {test_nan_count}")
            
            if train_nan_count > 0 or test_nan_count > 0:
                X_train.fillna(0, inplace=True)
                X_test.fillna(0, inplace=True)
                print("Replaced remaining NaN with 0")
            
            print(f"Final feature count: {X_train.shape[1]}")
            print(f"Final data shapes - training: {X_train.shape}, test: {X_test.shape}")
            
            gc.collect()
            
            return X_train, X_test, y_train, train_ids, test_ids
            
        except Exception as e:
            print(f"Error during data preprocessing: {e}")
            gc.collect()
            raise
    
    @timer
    def get_quick_processed_data(self):
        """Execute quick data preprocessing pipeline"""
        print("Starting quick data preprocessing pipeline")
        
        try:
            # 1. Quick data loading
            train_df, test_df = self.load_quick_data()
            
            # Save ID columns
            train_ids = train_df[Config.ID_COLUMN].copy()
            test_ids = test_df[Config.ID_COLUMN].copy()
            
            # Separate features and target
            X_train = train_df[self.feature_columns].copy()
            X_test = test_df[self.feature_columns].copy()
            y_train = train_df[Config.TARGET_COLUMN].copy()
            
            del train_df, test_df
            gc.collect()
            
            # 2. Create basic statistical features
            X_train, X_test = self.create_quick_statistical_features(X_train, X_test)
            
            # 3. Quick scaling
            X_train, X_test = self.scale_features(X_train, X_test, 'robust')
            
            # 4. Quick feature selection
            X_train, X_test = self.quick_feature_selection(X_train, X_test, y_train)
            
            # 5. Final validation
            train_nan_count = X_train.isna().sum().sum()
            test_nan_count = X_test.isna().sum().sum()
            
            if train_nan_count > 0 or test_nan_count > 0:
                X_train.fillna(0, inplace=True)
                X_test.fillna(0, inplace=True)
                print("Replaced remaining NaN with 0")
            
            print(f"Quick processing complete")
            print(f"Final data shapes - training: {X_train.shape}, test: {X_test.shape}")
            
            gc.collect()
            
            return X_train, X_test, y_train, train_ids, test_ids
            
        except Exception as e:
            print(f"Error during quick data preprocessing: {e}")
            gc.collect()
            raise
    
    def get_feature_importance(self):
        """Return feature importance"""
        return self.feature_importance
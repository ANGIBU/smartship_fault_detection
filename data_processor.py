# data_processor.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif, chi2, SelectFromModel, RFE
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
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
        self.class_distribution = None
        self.sensor_scalers = {}
        self.class_weights = None
        self.isolation_forests = {}
        
    @timer
    def load_data(self):
        """Load data with memory optimization"""
        print("Starting data loading with memory optimization")
        
        try:
            # Load with optimal dtypes
            dtype_dict = {col: 'float32' for col in self.feature_columns}
            dtype_dict[Config.ID_COLUMN] = 'object'
            dtype_dict[Config.TARGET_COLUMN] = 'int16'
            
            train_df = pd.read_csv(Config.TRAIN_FILE, dtype=dtype_dict)
            
            # Test file doesn't have target column
            test_dtype_dict = {col: 'float32' for col in self.feature_columns}
            test_dtype_dict[Config.ID_COLUMN] = 'object'
            test_df = pd.read_csv(Config.TEST_FILE, dtype=test_dtype_dict)
            
            print(f"Train data shape: {train_df.shape}")
            print(f"Test data shape: {test_df.shape}")
            
            if Config.TARGET_COLUMN in train_df.columns:
                # Analyze class distribution for class-balanced loss
                self.class_distribution = train_df[Config.TARGET_COLUMN].value_counts().sort_index()
                self._calculate_class_weights(train_df[Config.TARGET_COLUMN])
                
                print("Class distribution:")
                for class_id, count in self.class_distribution.items():
                    if class_id < 10:
                        print(f"  Class {class_id}: {count} samples")
                
                # Calculate imbalance ratio
                max_count = self.class_distribution.max()
                min_count = self.class_distribution.min()
                imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
                print(f"Class imbalance ratio: {imbalance_ratio:.2f}:1")
            
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
    
    def _calculate_class_weights(self, y_train):
        """Calculate class weights using effective number method"""
        if not Config.USE_CLASS_BALANCED_LOSS:
            return
        
        samples_per_class = []
        for class_id in range(Config.N_CLASSES):
            count = np.sum(y_train == class_id)
            samples_per_class.append(count)
        
        # Calculate effective number based weights
        self.class_weights = Config.get_effective_number_weights(samples_per_class)
        print(f"Class weights calculated using effective number method")
        
    @timer
    def load_quick_data(self):
        """Load data for quick mode"""
        print("Loading data for quick mode")
        
        try:
            # Load with memory-efficient dtypes
            dtype_dict = {col: 'float32' for col in self.feature_columns}
            dtype_dict[Config.ID_COLUMN] = 'object'
            dtype_dict[Config.TARGET_COLUMN] = 'int16'
            
            train_df = pd.read_csv(Config.TRAIN_FILE, dtype=dtype_dict)
            test_df = pd.read_csv(Config.TEST_FILE, dtype={col: 'float32' for col in self.feature_columns})
            
            # Stratified sampling to maintain class distribution
            if len(train_df) > Config.QUICK_SAMPLE_SIZE:
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
            
            # Basic data cleaning
            train_df, test_df = self._handle_data_issues(train_df, test_df)
            
            return train_df, test_df
            
        except Exception as e:
            print(f"Quick data loading failed: {e}")
            raise
    
    def _handle_data_issues(self, train_df, test_df):
        """Handle data issues with sensor-specific approach"""
        for col in self.feature_columns:
            if col in train_df.columns and col in test_df.columns:
                # Handle infinite values
                train_df[col] = train_df[col].replace([np.inf, -np.inf], np.nan)
                test_df[col] = test_df[col].replace([np.inf, -np.inf], np.nan)
                
                # Sensor-specific outlier detection and handling
                sensor_config = Config.get_sensor_specific_config(col)
                
                if sensor_config['outlier_method'] == 'iqr':
                    # IQR-based outlier detection
                    Q1 = train_df[col].quantile(0.25)
                    Q3 = train_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    # Cap outliers instead of removing
                    train_df[col] = train_df[col].clip(lower_bound, upper_bound)
                    test_df[col] = test_df[col].clip(lower_bound, upper_bound)
                
                elif sensor_config['outlier_method'] == 'zscore':
                    # Z-score based outlier detection
                    z_scores = np.abs((train_df[col] - train_df[col].mean()) / train_df[col].std())
                    outlier_threshold = 3.0
                    train_df.loc[z_scores > outlier_threshold, col] = train_df[col].median()
                
                # Handle missing values with cross-correlation based imputation
                if train_df[col].isnull().sum() > 0 or test_df[col].isnull().sum() > 0:
                    # Find most correlated sensors for imputation
                    correlations = train_df[self.feature_columns].corr()[col].abs().sort_values(ascending=False)
                    top_correlated = correlations.iloc[1:4].index.tolist()  # Skip self-correlation
                    
                    for df in [train_df, test_df]:
                        if df[col].isnull().sum() > 0:
                            # Use mean of top correlated sensors for imputation
                            impute_values = df[top_correlated].mean(axis=1)
                            df[col].fillna(impute_values, inplace=True)
                            
                            # If still NaN, use column median
                            if df[col].isnull().sum() > 0:
                                df[col].fillna(df[col].median(), inplace=True)
        
        return train_df, test_df
    
    @timer
    def create_statistical_features(self, X_train, X_test):
        """Create statistical features with sensor domain knowledge"""
        print("Creating statistical features with domain knowledge")
        
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
            
            # Domain-specific features for equipment monitoring
            if 'rms' in Config.STATISTICAL_FEATURES:
                # RMS (Root Mean Square) - important for vibration analysis
                df['sensor_rms'] = np.sqrt(np.mean(sensor_data**2, axis=1)).astype('float32')
            
            if 'crest_factor' in Config.STATISTICAL_FEATURES:
                # Crest Factor - peak to RMS ratio, important for fault detection
                sensor_max = df.get('sensor_max', np.max(sensor_data, axis=1))
                sensor_rms = df.get('sensor_rms', np.sqrt(np.mean(sensor_data**2, axis=1)))
                df['sensor_crest_factor'] = (sensor_max / (sensor_rms + 1e-8)).astype('float32')
        
        return X_train, X_test
    
    @timer
    def create_sensor_type_features(self, X_train, X_test):
        """Create sensor type specific features"""
        print("Creating sensor type specific features")
        
        for df in [X_train, X_test]:
            for sensor_type, sensor_list in Config.SENSOR_TYPES.items():
                available_sensors = [s for s in sensor_list if s in df.columns]
                
                if len(available_sensors) >= 2:
                    sensor_data = df[available_sensors].values
                    
                    # Type-specific statistics
                    df[f'{sensor_type}_mean'] = np.mean(sensor_data, axis=1).astype('float32')
                    df[f'{sensor_type}_std'] = np.std(sensor_data, axis=1).astype('float32')
                    df[f'{sensor_type}_range'] = (np.max(sensor_data, axis=1) - 
                                                 np.min(sensor_data, axis=1)).astype('float32')
                    
                    # Correlation within sensor type
                    if len(available_sensors) >= 3:
                        correlations = []
                        for i in range(len(sensor_data)):
                            if len(set(sensor_data[i])) > 1:  # Check for variance
                                corr_matrix = np.corrcoef(sensor_data[i:i+1].T)
                                if not np.isnan(corr_matrix).any():
                                    avg_corr = np.mean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])
                                    correlations.append(avg_corr)
                                else:
                                    correlations.append(0.0)
                            else:
                                correlations.append(0.0)
                        
                        df[f'{sensor_type}_avg_corr'] = np.array(correlations).astype('float32')
        
        return X_train, X_test
    
    @timer
    def create_domain_features(self, X_train, X_test):
        """Create domain-specific features for equipment monitoring"""
        print("Creating domain-specific features")
        
        for df in [X_train, X_test]:
            sensor_data = df[self.feature_columns].values
            
            # Equipment stability indicators
            std_vals = df.get('sensor_std', np.std(sensor_data, axis=1))
            mean_vals = df.get('sensor_mean', np.mean(sensor_data, axis=1))
            df['stability_index'] = (1 / (1 + std_vals / (np.abs(mean_vals) + 1e-8))).astype('float32')
            
            # Signal quality measures  
            df['signal_quality'] = (mean_vals / (std_vals + 1e-8)).astype('float32')
            
            # Sensor correlation features
            df['sensor_variance'] = np.var(sensor_data, axis=1).astype('float32')
            df['sensor_energy'] = np.sum(sensor_data**2, axis=1).astype('float32')
            
            # Percentile features
            df['sensor_p10'] = np.percentile(sensor_data, 10, axis=1).astype('float32')
            df['sensor_p90'] = np.percentile(sensor_data, 90, axis=1).astype('float32')
            
            # Cross-sensor relationships
            df['sensor_balance'] = (std_vals / (mean_vals + 1e-8)).astype('float32')
            
            # Anomaly indicators based on statistical measures
            df['statistical_anomaly'] = ((np.abs(mean_vals) > 3 * std_vals) | 
                                        (std_vals > 3 * np.mean(std_vals))).astype('int16')
        
        return X_train, X_test
    
    @timer
    def create_interaction_features(self, X_train, X_test):
        """Create interaction features between highly correlated sensors"""
        if not Config.CREATE_INTERACTION_FEATURES:
            return X_train, X_test
        
        print("Creating interaction features")
        
        # Find top correlated sensor pairs
        correlation_matrix = X_train[self.feature_columns].corr()
        
        # Get top N sensor pairs with highest correlation
        high_corr_pairs = []
        for i in range(len(self.feature_columns)):
            for j in range(i+1, len(self.feature_columns)):
                corr_val = abs(correlation_matrix.iloc[i, j])
                if corr_val > 0.7:  # High correlation threshold
                    high_corr_pairs.append((self.feature_columns[i], self.feature_columns[j], corr_val))
        
        # Sort by correlation and take top N
        high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
        top_pairs = high_corr_pairs[:Config.INTERACTION_TOP_N]
        
        print(f"Creating {len(top_pairs)} interaction features")
        
        for df in [X_train, X_test]:
            for sensor1, sensor2, corr_val in top_pairs:
                if sensor1 in df.columns and sensor2 in df.columns:
                    # Multiplicative interaction
                    df[f'{sensor1}_{sensor2}_mult'] = (df[sensor1] * df[sensor2]).astype('float32')
                    
                    # Ratio interaction (avoid division by zero)
                    df[f'{sensor1}_{sensor2}_ratio'] = (df[sensor1] / (df[sensor2] + 1e-8)).astype('float32')
                    
                    self.interaction_features.extend([f'{sensor1}_{sensor2}_mult', f'{sensor1}_{sensor2}_ratio'])
        
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
    def create_pca_features(self, X_train, X_test, n_components=6):
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
    def apply_class_balanced_loss_preparation(self, X_train, y_train):
        """Prepare data for class-balanced loss instead of traditional resampling"""
        if not Config.USE_CLASS_BALANCED_LOSS:
            return X_train, y_train
            
        print("Preparing data for class-balanced loss")
        
        # Calculate sample weights for training
        sample_weights = np.zeros(len(y_train))
        
        for class_id in range(Config.N_CLASSES):
            class_mask = (y_train == class_id)
            if np.any(class_mask):
                sample_weights[class_mask] = self.class_weights[class_id]
        
        # Store sample weights for use in training
        self.sample_weights = sample_weights
        
        print("Class-balanced loss preparation completed")
        print(f"Sample weights range: {sample_weights.min():.4f} - {sample_weights.max():.4f}")
        
        return X_train, y_train
    
    @timer
    def train_isolation_forests(self, X_train, y_train):
        """Train isolation forests for each class"""
        if 'isolation_forest' not in Config.CLASS_BALANCING_METHODS:
            return
        
        print("Training isolation forests for anomaly-based class balancing")
        
        for class_id in range(Config.N_CLASSES):
            class_mask = (y_train == class_id)
            if np.sum(class_mask) > 10:  # Need minimum samples
                class_data = X_train[class_mask]
                
                # Train isolation forest for this class
                iso_forest = IsolationForest(**Config.ISOLATION_FOREST_PARAMS)
                iso_forest.fit(class_data)
                
                self.isolation_forests[class_id] = iso_forest
                
                # Calculate anomaly scores for the class
                anomaly_scores = iso_forest.decision_function(class_data)
                print(f"Class {class_id}: {len(class_data)} samples, "
                      f"anomaly score range: {anomaly_scores.min():.3f} - {anomaly_scores.max():.3f}")
        
        print(f"Trained isolation forests for {len(self.isolation_forests)} classes")
    
    @timer
    def scale_features(self, X_train, X_test, method='sensor_specific'):
        """Feature scaling with sensor-specific approach"""
        print(f"Starting sensor-specific feature scaling")
        
        if method == 'sensor_specific':
            # Scale different sensor types with different methods
            for sensor_type, sensor_list in Config.SENSOR_TYPES.items():
                available_sensors = [s for s in sensor_list if s in X_train.columns]
                
                if available_sensors:
                    # Get sensor-specific configuration
                    sensor_config = Config.get_sensor_specific_config(available_sensors[0])
                    scaler_type = sensor_config['scaler']
                    
                    # Choose scaler based on sensor type
                    if scaler_type == 'robust':
                        scaler = RobustScaler()
                    elif scaler_type == 'minmax':
                        scaler = MinMaxScaler()
                    elif scaler_type == 'quantile':
                        scaler = QuantileTransformer(output_distribution='normal', 
                                                   random_state=Config.RANDOM_STATE)
                    else:
                        scaler = StandardScaler()
                    
                    # Fit and transform
                    X_train[available_sensors] = scaler.fit_transform(X_train[available_sensors])
                    X_test[available_sensors] = scaler.transform(X_test[available_sensors])
                    
                    self.sensor_scalers[sensor_type] = scaler
                    print(f"Scaled {len(available_sensors)} {sensor_type} sensors with {scaler_type}")
            
            # Scale derived features with standard scaler
            derived_features = [col for col in X_train.columns 
                              if col not in self.feature_columns and col not in [Config.ID_COLUMN]]
            
            if derived_features:
                self.scaler = StandardScaler()
                X_train[derived_features] = self.scaler.fit_transform(X_train[derived_features])
                X_test[derived_features] = self.scaler.transform(X_test[derived_features])
                print(f"Scaled {len(derived_features)} derived features with standard scaler")
        
        else:
            # Traditional single scaler approach
            if method == 'standard':
                self.scaler = StandardScaler()
            elif method == 'robust':
                self.scaler = RobustScaler()
            elif method == 'quantile':
                self.scaler = QuantileTransformer(
                    output_distribution='normal',
                    random_state=Config.RANDOM_STATE,
                    n_quantiles=min(1000, X_train.shape[0])
                )
            else:
                self.scaler = StandardScaler()
            
            # Scale only numeric columns
            numeric_columns = X_train.select_dtypes(include=[np.number]).columns.tolist()
            
            X_train[numeric_columns] = self.scaler.fit_transform(X_train[numeric_columns])
            X_test[numeric_columns] = self.scaler.transform(X_test[numeric_columns])
        
        # Maintain data types for memory efficiency
        for col in X_train.select_dtypes(include=[np.number]).columns:
            X_train[col] = X_train[col].astype('float32')
            X_test[col] = X_test[col].astype('float32')
        
        # Save scalers
        save_joblib(self.sensor_scalers if method == 'sensor_specific' else self.scaler, 
                   Config.SCALER_FILE)
        
        return X_train, X_test
    
    @timer
    def select_features(self, X_train, X_test, y_train, n_features=None):
        """Feature selection with multiple methods including RFE"""
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
            mi_selector = SelectKBest(mutual_info_classif, k=min(n_features, X_train.shape[1]))
            mi_selector.fit(X_train, y_train)
            mi_scores = mi_selector.scores_
            
            for i, col in enumerate(X_train.columns):
                if col not in feature_scores:
                    feature_scores[col] = 0
                feature_scores[col] += 0.35 * mi_scores[i] / (np.max(mi_scores) + 1e-8)
        except Exception as e:
            print(f"Mutual Information calculation failed: {e}")
        
        # F-test (weight: 0.25)
        try:
            f_selector = SelectKBest(f_classif, k=min(n_features, X_train.shape[1]))
            f_selector.fit(X_train, y_train)
            f_scores = f_selector.scores_
            
            for i, col in enumerate(X_train.columns):
                if col not in feature_scores:
                    feature_scores[col] = 0
                feature_scores[col] += 0.25 * f_scores[i] / (np.max(f_scores) + 1e-8)
        except Exception as e:
            print(f"F-test calculation failed: {e}")
        
        # Recursive Feature Elimination with Random Forest (weight: 0.4)
        try:
            rf_estimator = RandomForestClassifier(
                n_estimators=100, 
                random_state=Config.RANDOM_STATE,
                n_jobs=2,
                max_depth=8,
                class_weight='balanced'
            )
            
            rfe_selector = RFE(
                estimator=rf_estimator,
                n_features_to_select=min(n_features, X_train.shape[1]),
                step=0.1
            )
            rfe_selector.fit(X_train, y_train)
            
            rfe_ranking = rfe_selector.ranking_
            max_rank = np.max(rfe_ranking)
            
            for i, col in enumerate(X_train.columns):
                if col not in feature_scores:
                    feature_scores[col] = 0
                # Inverse ranking score (lower rank = higher score)
                feature_scores[col] += 0.4 * (max_rank - rfe_ranking[i] + 1) / max_rank
                
        except Exception as e:
            print(f"RFE calculation failed: {e}")
        
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
        pca_count = sum(1 for f in self.selected_features if f.startswith('pca_'))
        interaction_count = sum(1 for f in self.selected_features if '_mult' in f or '_ratio' in f)
        type_count = sum(1 for f in self.selected_features 
                        if any(sensor_type in f for sensor_type in Config.SENSOR_TYPES.keys()))
        other_count = (len(self.selected_features) - original_count - statistical_count - 
                      pca_count - interaction_count - type_count)
        
        print(f"Feature type distribution:")
        print(f"  Original sensors: {original_count}")
        print(f"  Statistical: {statistical_count}")
        print(f"  PCA: {pca_count}")
        print(f"  Interaction: {interaction_count}")
        print(f"  Sensor type: {type_count}")
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
    def get_processed_data(self, use_class_balanced_loss=True, scaling_method='sensor_specific'):
        """Execute complete data preprocessing pipeline with class-balanced approach"""
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
            
            # 3. Create sensor type specific features
            X_train, X_test = self.create_sensor_type_features(X_train, X_test)
            
            # 4. Create domain-specific features
            if Config.DOMAIN_FEATURES_ENABLED:
                X_train, X_test = self.create_domain_features(X_train, X_test)
            
            # 5. Create interaction features
            X_train, X_test = self.create_interaction_features(X_train, X_test)
            
            # 6. Create PCA features
            X_train, X_test = self.create_pca_features(X_train, X_test, n_components=Config.PCA_COMPONENTS)
            
            # 7. Scaling with sensor-specific approach
            X_train, X_test = self.scale_features(X_train, X_test, scaling_method)
            
            # 8. Feature selection
            X_train, X_test = self.select_features(X_train, X_test, y_train)
            
            # 9. Prepare for class-balanced loss instead of resampling
            if use_class_balanced_loss:
                X_train, y_train = self.apply_class_balanced_loss_preparation(X_train, y_train)
            
            # 10. Train isolation forests for anomaly detection
            self.train_isolation_forests(X_train, y_train)
            
            # 11. Final validation
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
            X_train, X_test = self.scale_features(X_train, X_test, 'standard')
            
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
    
    def get_class_weights(self):
        """Return calculated class weights"""
        return self.class_weights
    
    def get_sample_weights(self):
        """Return sample weights for class-balanced training"""
        return getattr(self, 'sample_weights', None)
    
    def get_isolation_forests(self):
        """Return trained isolation forests"""
        return self.isolation_forests
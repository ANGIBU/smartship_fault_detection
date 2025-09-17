# model_training.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import f1_score, make_scorer
import lightgbm as lgb
import xgboost as xgb
import optuna
import warnings
warnings.filterwarnings('ignore')

from config import Config
from utils import timer, calculate_macro_f1, print_classification_metrics, save_model, save_joblib, setup_logging

class ModelTraining:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.cv_scores = {}
        self.logger = setup_logging()
        self.scorer = make_scorer(f1_score, average='macro')
        
    @timer
    def train_lightgbm(self, X_train, y_train, X_val=None, y_val=None, params=None):
        """LightGBM 모델 훈련"""
        print("=== LightGBM 모델 훈련 시작 ===")
        
        if params is None:
            params = Config.LGBM_PARAMS.copy()
        else:
            params = params.copy()
        
        # validation 데이터가 있을 때만 early stopping 사용
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            eval_names = ['validation']
            early_stopping_rounds = params.pop('early_stopping_rounds', 100)
            
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train, y_train, 
                eval_set=eval_set, 
                eval_names=eval_names, 
                callbacks=[lgb.early_stopping(early_stopping_rounds)]
            )
        else:
            # validation 데이터가 없으면 early stopping 파라미터 제거
            params.pop('early_stopping_rounds', None)
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train)
        
        self.models['lightgbm'] = model
        self.logger.info("LightGBM 모델 훈련 완료")
        
        return model
    
    @timer
    def train_xgboost(self, X_train, y_train, X_val=None, y_val=None, params=None):
        """XGBoost 모델 훈련"""
        print("=== XGBoost 모델 훈련 시작 ===")
        
        if params is None:
            params = Config.XGB_PARAMS.copy()
        else:
            params = params.copy()
        
        # validation 데이터가 있을 때만 early stopping 사용
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            early_stopping_rounds = params.pop('early_stopping_rounds', 100)
            
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train, y_train, 
                eval_set=eval_set, 
                early_stopping_rounds=early_stopping_rounds,
                verbose=False
            )
        else:
            # validation 데이터가 없으면 early stopping 파라미터 제거
            params.pop('early_stopping_rounds', None)
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train)
        
        self.models['xgboost'] = model
        self.logger.info("XGBoost 모델 훈련 완료")
        
        return model
    
    @timer
    def train_random_forest(self, X_train, y_train, params=None):
        """Random Forest 모델 훈련"""
        print("=== Random Forest 모델 훈련 시작 ===")
        
        if params is None:
            params = Config.RF_PARAMS
        
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        self.models['random_forest'] = model
        self.logger.info("Random Forest 모델 훈련 완료")
        
        return model
    
    @timer
    def train_extra_trees(self, X_train, y_train, params=None):
        """Extra Trees 모델 훈련"""
        print("=== Extra Trees 모델 훈련 시작 ===")
        
        if params is None:
            params = Config.ET_PARAMS
        
        model = ExtraTreesClassifier(**params)
        model.fit(X_train, y_train)
        
        self.models['extra_trees'] = model
        self.logger.info("Extra Trees 모델 훈련 완료")
        
        return model
    
    @timer
    def train_gradient_boosting(self, X_train, y_train):
        """Gradient Boosting 모델 훈련"""
        print("=== Gradient Boosting 모델 훈련 시작 ===")
        
        params = {
            'n_estimators': 200,
            'learning_rate': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'random_state': Config.RANDOM_STATE
        }
        
        model = GradientBoostingClassifier(**params)
        model.fit(X_train, y_train)
        
        self.models['gradient_boosting'] = model
        self.logger.info("Gradient Boosting 모델 훈련 완료")
        
        return model
    
    @timer
    def train_svm(self, X_train, y_train):
        """SVM 모델 훈련"""
        print("=== SVM 모델 훈련 시작 ===")
        
        # 데이터 크기에 따라 파라미터 조정
        if len(X_train) > 10000:
            print("큰 데이터셋으로 인한 선형 SVM 사용")
            from sklearn.svm import LinearSVC
            model = LinearSVC(
                random_state=Config.RANDOM_STATE,
                max_iter=1000
            )
        else:
            model = SVC(
                kernel='rbf',
                probability=True,
                random_state=Config.RANDOM_STATE
            )
        
        model.fit(X_train, y_train)
        
        self.models['svm'] = model
        self.logger.info("SVM 모델 훈련 완료")
        
        return model
    
    @timer
    def train_knn(self, X_train, y_train):
        """KNN 모델 훈련"""
        print("=== KNN 모델 훈련 시작 ===")
        
        # 데이터 크기에 따라 k 값 조정
        k = min(5, len(X_train) // 100)
        k = max(3, k)
        
        model = KNeighborsClassifier(
            n_neighbors=k,
            n_jobs=Config.N_JOBS
        )
        
        model.fit(X_train, y_train)
        
        self.models['knn'] = model
        self.logger.info("KNN 모델 훈련 완료")
        
        return model
    
    @timer
    def train_naive_bayes(self, X_train, y_train):
        """Naive Bayes 모델 훈련"""
        print("=== Naive Bayes 모델 훈련 시작 ===")
        
        model = GaussianNB()
        model.fit(X_train, y_train)
        
        self.models['naive_bayes'] = model
        self.logger.info("Naive Bayes 모델 훈련 완료")
        
        return model
    
    @timer
    def train_logistic_regression(self, X_train, y_train):
        """Logistic Regression 모델 훈련"""
        print("=== Logistic Regression 모델 훈련 시작 ===")
        
        model = LogisticRegression(
            random_state=Config.RANDOM_STATE,
            max_iter=1000,
            n_jobs=Config.N_JOBS
        )
        
        model.fit(X_train, y_train)
        
        self.models['logistic_regression'] = model
        self.logger.info("Logistic Regression 모델 훈련 완료")
        
        return model
    
    @timer
    def hyperparameter_optimization_optuna(self, X_train, y_train, model_type='lightgbm', n_trials=100):
        """Optuna를 사용한 하이퍼파라미터 최적화"""
        print(f"=== {model_type} 하이퍼파라미터 최적화 시작 ===")
        
        def objective(trial):
            if model_type == 'lightgbm':
                params = {
                    'objective': 'multiclass',
                    'num_class': Config.N_CLASSES,
                    'metric': 'multi_logloss',
                    'boosting_type': 'gbdt',
                    'num_leaves': trial.suggest_int('num_leaves', 10, 200),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                    'verbose': -1,
                    'random_state': Config.RANDOM_STATE,
                    'n_estimators': 500,
                    'n_jobs': Config.N_JOBS
                }
                model = lgb.LGBMClassifier(**params)
                
            elif model_type == 'xgboost':
                params = {
                    'objective': 'multi:softprob',
                    'num_class': Config.N_CLASSES,
                    'eval_metric': 'mlogloss',
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                    'gamma': trial.suggest_float('gamma', 0, 5),
                    'random_state': Config.RANDOM_STATE,
                    'n_estimators': 500,
                    'n_jobs': Config.N_JOBS,
                    'tree_method': 'hist'
                }
                model = xgb.XGBClassifier(**params)
                
            elif model_type == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 5, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    'random_state': Config.RANDOM_STATE,
                    'n_jobs': Config.N_JOBS
                }
                model = RandomForestClassifier(**params)
            
            # 교차 검증
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=3, scoring=self.scorer, n_jobs=1
            )
            return cv_scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, timeout=Config.OPTUNA_TIMEOUT)
        
        print(f"최적 파라미터: {study.best_params}")
        print(f"최적 점수: {study.best_value:.4f}")
        
        return study.best_params, study.best_value
    
    @timer
    def hyperparameter_optimization_grid(self, X_train, y_train, model_type='random_forest'):
        """Grid Search를 사용한 하이퍼파라미터 최적화"""
        print(f"=== {model_type} Grid Search 시작 ===")
        
        if model_type == 'random_forest':
            model = RandomForestClassifier(random_state=Config.RANDOM_STATE, n_jobs=Config.N_JOBS)
            param_grid = {
                'n_estimators': [100, 300, 500],
                'max_depth': [10, 15, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif model_type == 'svm':
            model = SVC(random_state=Config.RANDOM_STATE, probability=True)
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
            }
        else:
            raise ValueError("지원하지 않는 모델 타입입니다.")
        
        grid_search = GridSearchCV(
            model, param_grid, 
            cv=3, scoring=self.scorer, 
            n_jobs=Config.N_JOBS, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"최적 파라미터: {grid_search.best_params_}")
        print(f"최적 점수: {grid_search.best_score_:.4f}")
        
        return grid_search.best_params_, grid_search.best_score_
    
    @timer
    def cross_validation(self, X_train, y_train, cv_folds=None):
        """교차 검증 수행"""
        if cv_folds is None:
            cv_folds = Config.CV_FOLDS
        
        print("=== 교차 검증 시작 ===")
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=Config.RANDOM_STATE)
        
        for model_name, model in self.models.items():
            print(f"\n{model_name} 교차 검증 중...")
            
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=skf, scoring=self.scorer, n_jobs=1
            )
            
            self.cv_scores[model_name] = {
                'scores': cv_scores,
                'mean': cv_scores.mean(),
                'std': cv_scores.std()
            }
            
            print(f"{model_name} CV 점수: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # 최고 성능 모델 선택
        if self.cv_scores:
            best_model_name = max(self.cv_scores.keys(), key=lambda x: self.cv_scores[x]['mean'])
            self.best_model = self.models[best_model_name]
            self.best_score = self.cv_scores[best_model_name]['mean']
            
            print(f"\n최고 성능 모델: {best_model_name}")
            print(f"최고 CV 점수: {self.best_score:.4f}")
        
        return self.cv_scores
    
    def _prepare_ensemble_estimators(self, estimator_list):
        """앙상블용 모델 파라미터 정리"""
        cleaned_estimators = []
        
        for name, model in estimator_list:
            # 모델을 복사하여 파라미터 정리
            if hasattr(model, 'get_params'):
                params = model.get_params()
                
                # early stopping 관련 파라미터 제거
                early_stopping_params = ['early_stopping_rounds', 'early_stopping_round']
                for param in early_stopping_params:
                    params.pop(param, None)
                
                # 새로운 모델 인스턴스 생성
                model_class = type(model)
                try:
                    new_model = model_class(**params)
                    cleaned_estimators.append((name, new_model))
                except Exception as e:
                    print(f"{name} 모델 파라미터 정리 중 오류: {e}")
                    # 기본 파라미터로 모델 생성
                    if 'lgb' in str(model_class):
                        new_model = lgb.LGBMClassifier(
                            objective='multiclass',
                            num_class=Config.N_CLASSES,
                            random_state=Config.RANDOM_STATE,
                            n_jobs=Config.N_JOBS,
                            verbose=-1
                        )
                    elif 'xgb' in str(model_class):
                        new_model = xgb.XGBClassifier(
                            objective='multi:softprob',
                            num_class=Config.N_CLASSES,
                            random_state=Config.RANDOM_STATE,
                            n_jobs=Config.N_JOBS,
                            tree_method='hist'
                        )
                    else:
                        new_model = model_class(random_state=Config.RANDOM_STATE)
                    cleaned_estimators.append((name, new_model))
            else:
                cleaned_estimators.append((name, model))
        
        return cleaned_estimators
    
    @timer
    def create_voting_ensemble(self, X_train, y_train, estimator_list=None):
        """투표 기반 앙상블 생성"""
        print("=== 투표 앙상블 생성 시작 ===")
        
        if estimator_list is None:
            # 성능이 좋은 상위 모델들 선택
            if len(self.models) < 2:
                print("앙상블을 위해 최소 2개의 모델이 필요합니다.")
                return None
            
            estimator_list = list(self.models.items())
        
        # 앙상블용 모델 파라미터 정리
        cleaned_estimators = self._prepare_ensemble_estimators(estimator_list)
        
        if len(cleaned_estimators) < 2:
            print("유효한 모델이 부족하여 앙상블을 생성할 수 없습니다.")
            return None
        
        # Hard voting과 Soft voting 모두 시도
        voting_classifiers = {}
        
        # Hard voting
        try:
            hard_voting = VotingClassifier(
                estimators=cleaned_estimators, 
                voting='hard',
                n_jobs=Config.N_JOBS
            )
            hard_voting.fit(X_train, y_train)
            voting_classifiers['hard_voting'] = hard_voting
            print("Hard voting 앙상블 생성 완료")
        except Exception as e:
            print(f"Hard voting 생성 실패: {e}")
        
        # Soft voting
        try:
            soft_voting = VotingClassifier(
                estimators=cleaned_estimators, 
                voting='soft',
                n_jobs=Config.N_JOBS
            )
            soft_voting.fit(X_train, y_train)
            voting_classifiers['soft_voting'] = soft_voting
            print("Soft voting 앙상블 생성 완료")
        except Exception as e:
            print(f"Soft voting 생성 실패: {e}")
        
        # 생성된 앙상블 모델들 추가
        self.models.update(voting_classifiers)
        
        return voting_classifiers
    
    @timer
    def create_stacking_ensemble(self, X_train, y_train, base_estimators=None):
        """스태킹 앙상블 생성"""
        print("=== 스태킹 앙상블 생성 시작 ===")
        
        if base_estimators is None:
            if len(self.models) < 2:
                print("스태킹을 위해 최소 2개의 모델이 필요합니다.")
                return None
            
            # 투표 기반 앙상블 제외
            base_estimators = [(name, model) for name, model in self.models.items() 
                             if 'voting' not in name and 'stacking' not in name and 'bagging' not in name]
        
        if len(base_estimators) < 2:
            print("스태킹을 위해 최소 2개의 기본 모델이 필요합니다.")
            return None
        
        # 스태킹용 모델 파라미터 정리
        cleaned_estimators = self._prepare_ensemble_estimators(base_estimators)
        
        # 메타 학습기 선택
        meta_classifier = LogisticRegression(
            random_state=Config.RANDOM_STATE, 
            max_iter=1000,
            n_jobs=Config.N_JOBS
        )
        
        try:
            stacking_ensemble = StackingClassifier(
                estimators=cleaned_estimators,
                final_estimator=meta_classifier,
                cv=Config.STACKING_CV,
                n_jobs=Config.N_JOBS,
                passthrough=False
            )
            
            stacking_ensemble.fit(X_train, y_train)
            self.models['stacking'] = stacking_ensemble
            
            print("스태킹 앙상블 생성 완료")
            return stacking_ensemble
        except Exception as e:
            print(f"스태킹 앙상블 생성 실패: {e}")
            return None
    
    @timer
    def create_bagging_ensemble(self, X_train, y_train):
        """배깅 앙상블 생성"""
        print("=== 배깅 앙상블 생성 시작 ===")
        
        # 다양한 기본 모델로 배깅 앙상블 생성
        base_models = [
            ('rf_bagging', BaggingClassifier(
                estimator=RandomForestClassifier(n_estimators=50, random_state=Config.RANDOM_STATE),
                n_estimators=10,
                random_state=Config.RANDOM_STATE,
                n_jobs=Config.N_JOBS
            )),
            ('et_bagging', BaggingClassifier(
                estimator=ExtraTreesClassifier(n_estimators=50, random_state=Config.RANDOM_STATE),
                n_estimators=10,
                random_state=Config.RANDOM_STATE,
                n_jobs=Config.N_JOBS
            ))
        ]
        
        for name, model in base_models:
            try:
                model.fit(X_train, y_train)
                self.models[name] = model
                print(f"{name} 생성 완료")
            except Exception as e:
                print(f"{name} 생성 실패: {e}")
        
        return base_models
    
    @timer
    def evaluate_model(self, model, X_val, y_val, model_name="Model"):
        """모델 평가"""
        print(f"=== {model_name} 평가 시작 ===")
        
        y_pred = model.predict(X_val)
        macro_f1 = print_classification_metrics(y_val, y_pred)
        
        return macro_f1
    
    @timer
    def feature_importance_analysis(self, model=None, feature_names=None):
        """피처 중요도 분석"""
        if model is None:
            model = self.best_model
        
        if model is None:
            print("분석할 모델이 없습니다.")
            return None
        
        print("=== 피처 중요도 분석 시작 ===")
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_).mean(axis=0)
        else:
            print("피처 중요도를 추출할 수 없는 모델입니다.")
            return None
        
        if feature_names is None and hasattr(model, 'feature_names_in_'):
            feature_names = model.feature_names_in_
        elif feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("상위 20개 중요 피처:")
        print(feature_importance_df.head(20))
        
        # 결과 저장
        from utils import save_results
        save_results(feature_importance_df, Config.FEATURE_IMPORTANCE_FILE)
        
        return feature_importance_df
    
    @timer
    def train_all_models(self, X_train, y_train, X_val=None, y_val=None, use_optimization=False, model_list=None):
        """모든 모델 훈련"""
        print("=== 전체 모델 훈련 시작 ===")
        
        if model_list is None:
            model_list = ['lightgbm', 'xgboost', 'random_forest', 'extra_trees']
        
        # 기본 모델들 훈련
        if 'lightgbm' in model_list:
            self.train_lightgbm(X_train, y_train, X_val, y_val)
        
        if 'xgboost' in model_list:
            self.train_xgboost(X_train, y_train, X_val, y_val)
        
        if 'random_forest' in model_list:
            self.train_random_forest(X_train, y_train)
        
        if 'extra_trees' in model_list:
            self.train_extra_trees(X_train, y_train)
        
        if 'gradient_boosting' in model_list:
            self.train_gradient_boosting(X_train, y_train)
        
        # 빠른 모델들 (필요시)
        if 'logistic_regression' in model_list:
            self.train_logistic_regression(X_train, y_train)
        
        if 'naive_bayes' in model_list:
            self.train_naive_bayes(X_train, y_train)
        
        # 하이퍼파라미터 최적화
        if use_optimization:
            print("하이퍼파라미터 최적화 수행 중...")
            best_lgb_params, best_lgb_score = self.hyperparameter_optimization_optuna(
                X_train, y_train, 'lightgbm', n_trials=Config.OPTUNA_TRIALS
            )
            
            # 최적 파라미터로 재훈련
            optimized_lgb = self.train_lightgbm(X_train, y_train, X_val, y_val, best_lgb_params)
            self.models['lightgbm_optimized'] = optimized_lgb
        
        # 앙상블 모델 생성
        if len(self.models) >= 2:
            self.create_voting_ensemble(X_train, y_train)
            self.create_stacking_ensemble(X_train, y_train)
            self.create_bagging_ensemble(X_train, y_train)
        
        # 교차 검증
        self.cross_validation(X_train, y_train)
        
        # 최고 모델 저장
        if self.best_model is not None:
            save_model(self.best_model, Config.MODEL_FILE)
            print(f"최고 모델 저장: {type(self.best_model).__name__}")
        
        # CV 결과 저장
        cv_results_df = pd.DataFrame(self.cv_scores).T
        cv_results_df.reset_index(inplace=True)
        cv_results_df.rename(columns={'index': 'model'}, inplace=True)
        
        from utils import save_results
        save_results(cv_results_df, Config.CV_RESULTS_FILE)
        
        print("=== 전체 모델 훈련 완료 ===")
        print(f"훈련된 모델 수: {len(self.models)}")
        
        return self.models, self.best_model
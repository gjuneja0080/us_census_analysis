import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
import lightgbm as lgb

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold)
from sklearn.metrics import (f1_score, roc_auc_score, precision_score, recall_score, classification_report, confusion_matrix, precision_recall_curve, confusion_matrix)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks


class ModelTrainer:
    def __init__(self, learn_df, test_df, target_col="total_person_income"):
        """
        learn_df, test_df: df for training and testing
        target_col: total person income
        """
        self.learn_df = learn_df
        self.test_df = test_df
        self.target_col = target_col
        self.searches = {} 
        self.best_estimators = {}  

    def get_train_test(self):
        """
        Splits final data into tes and training sets
        """
        X_train = self.learn_df.drop(self.target_col, axis=1)
        y_train = self.learn_df[self.target_col]
        X_test = self.test_df.drop(self.target_col, axis=1)
        y_test = self.test_df[self.target_col]
        return X_train, X_test, y_train, y_test

    def build_preprocessor(self, selected_numeric):
        """
        Build polynomial and standard scaler pipeline 
        """
        numeric_transformer = Pipeline(steps=[('poly', PolynomialFeatures(degree=2, interaction_only=True)), ('scaler', StandardScaler())])
        preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, selected_numeric)], remainder='passthrough')
        return preprocessor

    def transform_with_preprocessor(self, preprocessor, X_train, X_test):
        """
        Fit on X_trainand  then transform X_test
        """
        X_train_t = preprocessor.fit_transform(X_train)
        X_test_t = preprocessor.transform(X_test)
        return X_train_t, X_test_t

    def smote_tomek_resample(self, X_train, y_train):
        """
        Applies SMOTE + Tomek with custom parameters 
        """
        smote = SMOTE(sampling_strategy=0.5, k_neighbors=10, random_state=42)
        tomek = TomekLinks(sampling_strategy='all')
        smote_tomek = SMOTETomek(smote=smote, tomek=tomek, random_state=42)
        X_train_res, y_train_res = smote_tomek.fit_resample(X_train, y_train)
        return X_train_res, y_train_res

    def train_xgboost(self, X_train, y_train):
        """
        RandomizedSearchCV for XGBoost
        """
        xgb_params = {
            'n_estimators': [100, 200, 500],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 6, 10, 15],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'lambda': [0.1, 1.0, 10.0],
            'alpha': [0.1, 1.0, 10.0]
        }

        model = XGBClassifier(eval_metric='logloss', scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]))
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        search = RandomizedSearchCV(model, xgb_params, n_iter=50, scoring='roc_auc', cv=cv, verbose=1, random_state=42, n_jobs=-1)
        search.fit(X_train, y_train)
        self.searches['xgb'] = search
        self.best_estimators['xgb'] = search.best_estimator_
        print("XGBoost Best parameters:", search.best_params_)

    def train_lightgbm(self, X_train, y_train):
        """
        RandomizedSearchCV for LightGBM
        """
        lgb_est = lgb.LGBMClassifier()
        lgb_params = {
            'n_estimators': [100, 200, 500],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 6, 10, 15, -1], 
            'num_leaves': [10, 20, 31, 63, 127],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_lambda': [0.1, 1.0, 10.0],
            'reg_alpha': [0.1, 1.0, 10.0]
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        search = RandomizedSearchCV(lgb_est, lgb_params, n_iter=50, scoring='roc_auc', cv=cv, verbose=0, random_state=42, n_jobs=-1)
        search.fit(X_train, y_train)
        self.searches['lgb'] = search
        self.best_estimators['lgb'] = search.best_estimator_
        print("LightGBM Best parameters:", search.best_params_)

    def train_logistic_regression(self, X_train, y_train):
        """
        RandomizedSearchCV for Logistic Regression
        """
        lr = LogisticRegression(solver='liblinear', max_iter=1000, random_state=42)
        lr_params = {'C': [0.01, 0.1, 1.0, 10.0], 'penalty': ['l1', 'l2']}
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        search = RandomizedSearchCV(lr, lr_params, n_iter=6, scoring='roc_auc', cv=cv, verbose=1, random_state=42, n_jobs=-1)
        search.fit(X_train, y_train)
        self.searches['lr'] = search
        self.best_estimators['lr'] = search.best_estimator_
        print("Logistic Regression Best parameters:", search.best_params_)


    def evaluate_model(self, model_key, X_test, y_test, threshold=None):
        """
        assesses the model stored in self.best_estimators[model_key]
        """
        estimator = self.best_estimators.get(model_key, None)
        if estimator is None:
            print(f"No trained model found for key: {model_key}")
            return
        probs = estimator.predict_proba(X_test)[:, 1]
        if threshold is not None:
            preds = (probs > threshold).astype(int)
        else:
            preds = estimator.predict(X_test)

        report = classification_report(y_test, preds, target_names=['Less than 50k', 'Greater than 50k'])
        precision = precision_score(y_test, preds)
        recall = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        roc_auc = roc_auc_score(y_test, probs)

        print(f"Model: {model_key}")
        print(report)
        print("Evaluation Metrics:")
        print(pd.Series({ 'precision': precision, 'recall': recall, 'f1': f1, 'roc_auc': roc_auc}))

        print("Confusion Matrix:")

        cm = confusion_matrix(y_test, preds, normalize='true')
        labels = [['TN\n{:.2f}'.format(cm[0, 0]), 'FP\n{:.2f}'.format(cm[0, 1])],
          ['FN\n{:.2f}'.format(cm[1, 0]), 'TP\n{:.2f}'.format(cm[1, 1])]]
        
        sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=False,
            xticklabels=['Predicted <50k', 'Predicted >50k'],
            yticklabels=['Actual <50k', 'Actual >50k'])

        plt.xlabel('Predicted Labels')
        plt.ylabel('Actual Labels')

        plt.show()

    def print_feature_imp(self, model_key, preprocessor):
        model = self.best_estimators[model_key]

        feature_importances = model.feature_importances_
        num_transformed_features = preprocessor.get_feature_names_out()
        fi_df = pd.DataFrame({
            'Feature': num_transformed_features,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)

        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(fi_df)
        

    def feature_importance_filter_and_refit(self, model_key, preprocessor, X_train, y_train, X_test, y_test, importance_threshold=0.01, refit_entire_search=False):
        """
        it gets feature importances from the best estimator and then identifies features above importance_threshold.
        then it reruns either RandomizedSearchCV or fits the data with reduced features on existing best estimator
        """
        if model_key not in self.best_estimators:
            print(f"No trained model found for key: {model_key}")
            return
        
        model = self.best_estimators[model_key]

        feature_importances = model.feature_importances_

        num_transformed_features = preprocessor.get_feature_names_out()
        fi_df = pd.DataFrame({
            'Feature': num_transformed_features,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)

        # print(f"[{model_key.upper()}] Feature Importances:\n", fi_df)

        high_importance_features = fi_df[fi_df['Importance'] > importance_threshold]['Feature']

        X_train_df = pd.DataFrame(X_train, columns=num_transformed_features)
        X_test_df = pd.DataFrame(X_test, columns=num_transformed_features)

        X_train_filt = X_train_df[high_importance_features]
        X_test_filt = X_test_df[high_importance_features]

        if refit_entire_search:
            print(f"Resrunning RandomizedSearchCV for model {model_key} on filtered features")
            self.searches[model_key].fit(X_train_filt, y_train)
            self.best_estimators[model_key] = self.searches[model_key].best_estimator_
        else:
            print(f"Refitting the best estimator for model {model_key} on filtered features")
            model.fit(X_train_filt, y_train)

        print(f"\nEvaluating after feature importance filtering for model: {model_key}")
        self.evaluate_model(model_key, X_test_filt, y_test)

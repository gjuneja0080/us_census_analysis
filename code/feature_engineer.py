import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import mutual_info_classif


class FeatureEngineer:
    def __init__(self, learn_df, test_df):
        self.learn_df = learn_df
        self.test_df = test_df
        self.target_col = "total_person_income"

    def create_new_features(self):
        """
        Creates the new features from your original code:
          capital_losses_to_age
          capital_gains_to_hours
          wage_per_age
          has_capital_gains
          is_full_time
          income_potential
        """
        self.learn_df['capital_losses_to_age'] = self.learn_df['capital losses'] / (self.learn_df['age'] + 1)
        self.test_df['capital_losses_to_age'] = self.test_df['capital losses'] / (self.test_df['age'] + 1)

        self.learn_df['capital_gains_to_hours'] = self.learn_df['capital gains'] / (self.learn_df['wage per hour'] + 1)
        self.test_df['capital_gains_to_hours'] = self.test_df['capital gains'] / (self.test_df['wage per hour'] + 1)

        self.learn_df['wage_per_age'] = self.learn_df['wage per hour'] / (self.learn_df['age'] + 1)
        self.test_df['wage_per_age'] = self.test_df['wage per hour'] / (self.test_df['age'] + 1)

        self.learn_df['has_capital_gains'] = (self.learn_df['capital gains'] > 0).astype(int)
        self.test_df['has_capital_gains'] = (self.test_df['capital gains'] > 0).astype(int)

        self.learn_df['is_full_time'] = (self.learn_df['weeks worked in year'] >= 40).astype(int)
        self.test_df['is_full_time'] = (self.test_df['weeks worked in year'] >= 40).astype(int)

        self.learn_df['income_potential'] = self.learn_df['age'] * self.learn_df['weeks worked in year'] * self.learn_df['wage per hour']
        self.test_df['income_potential'] = self.test_df['age'] * self.test_df['weeks worked in year'] * self.test_df['wage per hour']

    def binarize_target(self):
        """
        Converts total person income to binary 1 and 0
        """
        self.learn_df[self.target_col] = self.learn_df[self.target_col].astype('int64')
        self.test_df[self.target_col] = self.test_df[self.target_col].astype('int64')

        self.learn_df[self.target_col] = self.learn_df[self.target_col].apply(lambda x: 1 if x > 0 else 0)
        self.test_df[self.target_col] = self.test_df[self.target_col].apply(lambda x: 1 if x > 0 else 0)

    def group_rare_countries(self):
        """
        Groups rare categories
        """
        for col in ['country of birth father', 'country of birth mother', 
                    'detailed household and family stat', 'major occupation code']:
            top_n = self.learn_df[col].value_counts().nlargest(20).index
            self.learn_df[col] = self.learn_df[col].apply(lambda x: x if x in top_n else 'Other')
            self.test_df[col] = self.test_df[col].apply(lambda x: x if x in top_n else 'Other')

    def log_transform(self):
        """
        log transforms
        """
        for col in ['capital gains', 'capital losses', 'dividends from stocks', 'wage per hour']:
            self.learn_df[f'{col}_log'] = np.log1p(self.learn_df[col])
            self.learn_df.drop(col, axis=1, inplace=True)

            self.test_df[f'{col}_log'] = np.log1p(self.test_df[col])
            self.test_df.drop(col, axis=1, inplace=True)


    def label_or_target_encode(self, cols_to_analyze, threshold=10, n_splits=5):
        """
        splits columns into label-encoded  or target-encoded 
        """

        card = self.learn_df[cols_to_analyze].nunique()
        low_card_cols = card[card <= threshold].index.tolist()
        high_card_cols = card[card > threshold].index.tolist()

        for col in low_card_cols:
            le = LabelEncoder()
            self.learn_df[col] = le.fit_transform(self.learn_df[col])
            self.test_df[col] = le.transform(self.test_df[col])

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        encoder = TargetEncoder()

        for col in high_card_cols:
            self.learn_df[col] = self.learn_df[col].astype(str)
            self.test_df[col] = self.test_df[col].astype(str)

            self.learn_df[col + '_te'] = np.zeros(self.learn_df.shape[0])

            for train_idx, val_idx in skf.split(self.learn_df, self.learn_df[self.target_col]):
                encoder.fit(self.learn_df.iloc[train_idx][col], self.learn_df.iloc[train_idx][self.target_col])

                self.learn_df.iloc[val_idx, self.learn_df.columns.get_loc(col + '_te')] = encoder.transform(self.learn_df.iloc[val_idx][col])

            self.test_df[col + '_te'] = encoder.transform(self.test_df[col])

            self.learn_df.drop(columns=col, inplace=True)
            self.test_df.drop(columns=col, inplace=True)

        print("Low Cardinality Columns (Label Encoded):", low_card_cols)
        print("High Cardinality Columns (Target Encoded):", high_card_cols)

    def ensure_numeric(self):
        """
        Removes columns that have non-numeric content
        """
        invalid_values = self.learn_df.applymap(lambda x: isinstance(x, str))
        non_numeric_cols = invalid_values.any(axis=0)

        self.learn_df.drop(columns=self.learn_df.columns[non_numeric_cols], inplace=True)
        self.test_df.drop(columns=self.test_df.columns[non_numeric_cols], inplace=True)

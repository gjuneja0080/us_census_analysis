import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2_contingency

class DataCleaner:
    def __init__(self, learn_df, test_df):
        """
        Initialize with raw dataframes
        """
        self.learn_df = learn_df
        self.test_df = test_df
        self.encoders = {}
        self.imputer = None

    def check_duplicates(self, drop_target_weight=False):
        """
        Prints duplicates in learn/test
        """
        if drop_target_weight:
            learn_dup = self.learn_df.drop(["total_person_income", "instance weight"], axis=1).duplicated().sum()
            test_dup = self.test_df.drop(["total_person_income", "instance weight"], axis=1).duplicated().sum()
        else:
            learn_dup = self.learn_df.duplicated().sum()
            test_dup = self.test_df.duplicated().sum()

        print(f"Duplicates in learn: {learn_dup}, Duplicates in test: {test_dup}")

    @staticmethod
    def aggregate_instance_weights(df):
        """
        Aggregates rows that are duplicates and sums instance weights
        """
        group_cols = df.columns.difference(['instance weight']).tolist()
        df = df.groupby(group_cols, as_index=False, dropna=False).agg({'instance weight': 'sum'})
        return df

    @staticmethod
    def resolve_label_conflicts(df):
        """
        weighted majority vote if there's a conflict of target labels
        """
        group_cols = df.columns.difference(['total_person_income', 'instance weight']).tolist()

        def weighted_majority(group):
            votes = group.groupby('total_person_income')['instance weight'].sum()
            return votes.idxmax()

        resolved_labels = df.groupby(group_cols, dropna=False).apply(weighted_majority).reset_index(name='resolved_label')
        df = df.merge(resolved_labels, on=group_cols, how='left')
        df['total_person_income'] = df['resolved_label']
        df = df.drop(columns=['resolved_label'])
        return df.drop_duplicates()

    def remove_duplicates_and_resolve_labels(self):
        """
        Aggregates duplicates for both learn_df and test_dfthen resolves label conflicts.
        """
        self.learn_df = self.aggregate_instance_weights(self.learn_df)
        self.learn_df = self.resolve_label_conflicts(self.learn_df)
        

        self.test_df = self.aggregate_instance_weights(self.test_df)
        self.test_df = self.resolve_label_conflicts(self.test_df)
        


    def drop_high_missing_cols(self, threshold=10):
        """
        Drops columns with > threshold% missing values
        """
        learn_test_combined = pd.concat([self.learn_df, self.test_df], axis=0, ignore_index=True)
        
        missing_counts = learn_test_combined.isnull().sum()
        missing_percent = (missing_counts / learn_test_combined.shape[0]) * 100
        
        high_missing_cols = missing_percent[missing_percent > threshold].index.tolist()
        
        # print(high_missing_cols, missing_counts, missing_percent)
        
        missing_info = pd.DataFrame({'Column': high_missing_cols, 'Missing Count': missing_counts[high_missing_cols].values,'Missing Percentage': missing_percent[high_missing_cols].values})
        
        self.learn_df.drop(columns=high_missing_cols, inplace=True, errors='ignore')
        self.test_df.drop(columns=high_missing_cols, inplace=True, errors='ignore')
        
        return missing_info

    def encode_and_impute(self):
        """
         Label-encode categorical columns
         KNN-impute both learn & test 
         decode them back to original string labels
        """

        cat_cols = self.learn_df.select_dtypes(include=['object']).columns.tolist()

        for col in cat_cols:
            le = LabelEncoder()
            self.learn_df[col] = self.learn_df[col].astype(str)
            valid_mask = self.learn_df[col].notnull()
            self.learn_df.loc[valid_mask, col] = le.fit_transform(self.learn_df.loc[valid_mask, col])
            self.learn_df[col] = self.learn_df[col].astype(float)
            self.encoders[col] = le
        self.imputer = KNNImputer(n_neighbors=3)
        learn_imputed = self.imputer.fit_transform(self.learn_df)
        self.learn_df[:] = learn_imputed


        for col in cat_cols:
            self.learn_df[col] = np.round(self.learn_df[col]).astype(int)
            self.learn_df[col] = self.encoders[col].inverse_transform(self.learn_df[col])

        for col in cat_cols:
            valid_mask = self.test_df[col].notnull()
            self.test_df.loc[valid_mask, col] = self.encoders[col].transform(self.test_df.loc[valid_mask, col])
            self.test_df[col] = self.test_df[col].astype(float)
        test_imputed = self.imputer.transform(self.test_df)
        self.test_df[:] = test_imputed
        for col in cat_cols:
            self.test_df[col] = np.round(self.test_df[col]).astype(int)
            self.test_df[col] = self.encoders[col].inverse_transform(self.test_df[col])

    @staticmethod
    def cramers_v(confusion_matrix):
        """
        Compute cramers V for nominal association
        """
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        k = min(confusion_matrix.shape) - 1
        return np.sqrt(chi2 / (n * k))

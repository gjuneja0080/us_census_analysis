import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu, chi2_contingency

class EDA:
    def __init__(self, learn_df: pd.DataFrame, test_df: pd.DataFrame):
        self.learn_df = learn_df.copy()
        self.test_df = test_df.copy()

    def distribution_of_target(self):
        """
        Plots improved distribution of 'total_person_income'
        """
        income_counts = self.learn_df['total_person_income'].value_counts()
        income_percent = (income_counts / income_counts.sum()) * 100
        income_summary = pd.DataFrame({'Count': income_counts, 'Percentage': income_percent})
        print(income_summary)

        plt.figure(figsize=(8, 6))
        ax = sns.countplot(data=self.learn_df, x='total_person_income', palette='pastel')

        plt.title('Distribution of Total Person Income')
        plt.xlabel('Income Category')
        plt.ylabel('Count')
        plt.show()

    def eda_continuous_cols(self, continuous_cols):
        """
        Prints summary stats
        """
        desc = self.learn_df[continuous_cols].describe()
        return desc

    def violin_plots(self, continuous_cols):
        plt.figure(figsize=(16, 12))
        for i, col in enumerate(continuous_cols, 1):
            plt.subplot(4, 2, i)
            sns.violinplot(data=self.learn_df, x=col, color="skyblue", inner="quartile")
            plt.title(f"Violin Plot: {col}", fontsize=12, fontweight='bold')
            plt.xlabel("")
            plt.ylabel("Density")

        plt.tight_layout()
        plt.show()

    def correlation_heatmap(self, continuous_cols):
        """
        Builds correlation heatmap
        """
        corr_matrix = self.learn_df[continuous_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='viridis')
        plt.title('Correlation Heatmap')
        plt.show()

    def pairwise_relationship(self, continuous_cols):
        """
        Builds pair plot
        """
        sns.pairplot(self.learn_df[continuous_cols], diag_kind="kde", plot_kws={"alpha": 0.5})
        plt.suptitle("Pairwise Relationships and Distributions", y=1.02)
        plt.show()

    def fix_target_income_column(self):
        """
        Fixes total person income it has weird symbols
        """
        self.learn_df['total_person_income'] = self.learn_df['total_person_income'].replace({'- 50000.': -50000,'50000+.': 50000}).astype('int64')
        self.test_df['total_person_income'] = self.test_df['total_person_income'].replace({'- 50000.': -50000,'50000+.': 50000}).astype('int64')

    def strip_plots_vs_target(self, continuous_cols):
        """
        scatter plots of each continuous variable vs. 'total_person_income'
        """
        num_cols = len(continuous_cols)
        rows = 2
        cols = 4

        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(20, 8))
        axes = axes.flatten() 

        for i, col in enumerate(continuous_cols):
            sns.stripplot(data=self.learn_df, x='total_person_income', y=col, jitter=True, ax=axes[i], color='#48a47c')
            axes[i].set_title(f'{col} vs. Total Person Income')

        for j in range(num_cols, rows*cols):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3, wspace=0.5)
        plt.show()

    def split_hist_plots(self, continuous_cols):
        """
        Histogram plots for total_person_income
        """
        num_cols = len(continuous_cols)
        rows = len(continuous_cols)
        cols = 2  

        fig, axes = plt.subplots(nrows=rows, ncols=2, figsize=(15, 5 * rows))
        axes = axes.flatten()

  
        for i, col in enumerate(continuous_cols):
            sns.histplot(data=self.learn_df[self.learn_df['total_person_income'] == '- 50000.'],
                        x=col, bins=50, kde=False, ax=axes[2 * i], color='#406484')
            axes[2 * i].set_title(f"{col} vs Total Person Income (<50K)")
            axes[2 * i].set_xlabel(col)
            axes[2 * i].set_ylabel('Frequency')
            sns.histplot(data=self.learn_df[self.learn_df['total_person_income'] == '50000+.'],
                        x=col, bins=50, kde=False, ax=axes[2 * i + 1], color='#48a47c')
            axes[2 * i + 1].set_title(f"{col} vs Total Person Income (≥50K)")
            axes[2 * i + 1].set_xlabel(col)
            axes[2 * i + 1].set_ylabel('Frequency')

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.2, wspace=0.2)
        plt.show()

    def mannwhitney(self, continuous_cols):
        """
        Mann-Whitney for each continuous col 
        """
        for col in continuous_cols:
            group_neg = self.learn_df[self.learn_df['total_person_income'] == -50000][col].dropna()
            group_pos = self.learn_df[self.learn_df['total_person_income'] == 50000][col].dropna()
            if len(group_neg) > 0 and len(group_pos) > 0:
                stat, p = mannwhitneyu(group_neg, group_pos, alternative='two-sided')
                print(f'{col}: Mann-Whitney U test statistic = {stat:.4f}, p-value = {p:.4f}')

    def nominal_feature_analysis(self, nominal_cols):
        """
        Computes chi2 test & cramer's v for nominal cols
        """
        results = []
        for col in nominal_cols:
            contingency_table = pd.crosstab(self.learn_df[col], self.learn_df['total_person_income'])
            chi2, p, dof, expected = chi2_contingency(contingency_table)
            v = self.cramers_v(contingency_table)
            results.append({'Column': col, 'Chi2_p_value': p, "Cramer's V": v})

        return pd.DataFrame(results).sort_values(by="Cramer's V", ascending=False).reset_index(drop=True)

    @staticmethod
    def cramers_v(confusion_matrix):
        """
        cramers v
        """
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        k = min(confusion_matrix.shape) - 1
        return np.sqrt(chi2 / (n * k))


    def bar_plots_of_nominal(self, columns):
        """
        Horizontal bar plots for nominal columns
        """
        for col in columns:
            counts = self.learn_df[col].value_counts().sort_values(ascending=True)
            plt.figure(figsize=(15, min(1 + 0.5 * len(counts), 20)))
            plt.barh(counts.index.astype(str), counts.values, color='blue')
            plt.title(f'Distribution of {col}')
            plt.xlabel('Count')
            plt.ylabel(col)
            plt.tight_layout(pad=2)
            plt.show()

    def stacked_bar_proportions(self, columns):
        """
        Stacked bar of proportion of the total person income
        """
        for col in columns:
            crosstab = pd.crosstab(self.learn_df[col], self.learn_df['total_person_income'], normalize='index')
            ax = crosstab.plot(kind='bar', stacked=True, figsize=(10, 5), color=['#406484', '#48a47c'])
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')  # Rotate 45 degrees with alignment
            plt.title(f'Proportion of Income by {col}')
            plt.ylabel('Proportion')
            plt.xlabel(col)
            plt.tight_layout()
            plt.show()

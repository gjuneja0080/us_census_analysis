import pandas as pd

class DataLoader:
    def __init__(self, metadata_path: str, learn_path: str, test_path: str):
        """
        Initializes file paths.
        """
        self.metadata_path = metadata_path
        self.learn_path = learn_path
        self.test_path = test_path
        self.col_names = []

    def load_metadata(self):
        """
        Reads the metadata file to extract column names
        """
        with open(self.metadata_path, "r") as f:
            for line in f:
                if ":" in line and line.strip()[0].isalpha():
                    col_name = line.split(":")[0].strip()
                    self.col_names.append(col_name)

        if "instance weight" not in self.col_names:
            self.col_names.insert(24, "instance weight")
        if "total_person_income" not in self.col_names:
            self.col_names.append("total_person_income")
        self.col_names = list(dict.fromkeys(self.col_names))

    def load_dataframes(self):
        """
        Uses self.col_names to read the CSV files into pandas DataFrames.
        """
        learn_df = pd.read_csv(self.learn_path, header=None, sep=", ", na_values="?", names=self.col_names)
        test_df = pd.read_csv(self.test_path, header=None, sep=", ", na_values="?", names=self.col_names)
        return learn_df, test_df

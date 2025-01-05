<b>Overview</b>

This project implements a machine learning pipeline for a binary classification problem that predicts whethe people earn more than or less than $50,000 using US Census Data. 
The pipeline includes data preprocessing, exploratory data analysis (EDA), feature engineering and training multiple models with hyperparameter tuning and performance evaluation. 


```
├── data_loader.py               # Handles data loading from raw files
├── data_cleaner.py              # Performs data cleaning and preprocessing
├── eda.py                       # Generates exploratory data analysis reports
├── feature_engineer.py          # Applies feature engineering techniques
├── model_trainer.py             # Trains ML models with hyperparameter tuning
├── requirements.txt             # Python dependencies for pip
├── environment.yml              # Python dependencies for conda
├── README.md                    # Documentation
└── datasets/
    ├── train.csv                # Training dataset
    ├── test.csv                 # Testing dataset
    └── metadata.txt             # Metadata file with column definitions
```

<b>Dependencies</b>
For Pip Installations:
```
pip install -r requirements.txt
```

For Conda Environment:
```
conda env create -f environment.yml
conda activate ml_env
```

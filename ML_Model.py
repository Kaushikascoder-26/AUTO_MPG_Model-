import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


##functions

acc_ix, hpower_ix, cyl_ix = 3 , 5 , 1

class CustomAttrAdder(BaseEstimator, TransformerMixin):
    def __init__(self, acc_on_power=True): # no *args or **kargs
        self.acc_on_power = acc_on_power
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        acc_on_cyl = X[:, acc_ix] / X[:, cyl_ix]
        if self.acc_on_power:
            acc_on_power = X[:, acc_ix] / X[:, hpower_ix]
            return np.c_[X, acc_on_power, acc_on_cyl]
        
        return np.c_[X, acc_on_cyl]


def num_pipeline_transformer(data):
    numerics = ['float64', 'int64']
    num_attrs = data.select_dtypes(include=numerics).columns
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attrs_adder', CustomAttrAdder()),
        ('std_scaler', StandardScaler()),
    ])
    prepared_data = num_pipeline.fit_transform(data[num_attrs])
    return prepared_data

# ml_model.py
# ... (previous code)

def pipeline_transformer(data, fit_transform=True):
    num_attrs = data.select_dtypes(include=['float64', 'int64']).columns
    cat_attrs = ['Origin']

    num_pipeline = Pipeline([
        ('num_imputer', SimpleImputer(strategy="median")),
        ('num_attrs_adder', CustomAttrAdder()),
        ('std_scaler', StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ('cat_imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot_encoder', OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')),
    ])

    full_pipeline = ColumnTransformer([
        ('num', num_pipeline, num_attrs),
        ('cat', cat_pipeline, cat_attrs),
    ])

    if fit_transform:
        prepared_data = full_pipeline.fit_transform(data)
    else:
        prepared_data = full_pipeline.transform(data)

    num_col_names = num_attrs.tolist()
    cat_encoder = full_pipeline.named_transformers_['cat']['onehot_encoder']
    cat_col_names = cat_encoder.get_feature_names(cat_attrs)

    all_col_names = num_col_names + cat_col_names.tolist()
    prepared_df = pd.DataFrame(prepared_data, columns=all_col_names)

    return prepared_df


def preprocess_origin_cols(df):
    df["Origin"] = df["Origin"].map({1: "India", 2: "USA", 3: "Germany"})
    return df

# ml_model.py
# ... (previous code)

def predict_mpg(config, model):
    if type(config) == dict:
        df = pd.DataFrame(config)
    else:
        df = config
    
    preproc_df = preprocess_origin_cols(df)
    prepared_df = pipeline_transformer(preproc_df, fit_transform=False)

    predictions = model.predict(prepared_df)

    return predictions[0]

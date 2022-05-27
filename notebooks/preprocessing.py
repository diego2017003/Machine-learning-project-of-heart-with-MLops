import logging
from multiprocessing import Value
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn_pandas import DataFrameMapper
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


def get_numerical_categorical_features(data: pd.DataFrame):
    numerical_features = []
    categorical_features = []
    for i in data.columns:
        if data[i].dtypes == "float64":
            numerical_features.append(i)
        else:
            categorical_features.append(i)
    return numerical_features, categorical_features


def split_categorical_binary_features(data: pd.DataFrame):
    binary_features = []
    categorical_features = []
    for i in data.columns:
        if len(data[i].unique()) == 2:
            binary_features.append(i)
        else:
            categorical_features.append(i)
    return binary_features, categorical_features


def treat_numerical_data(data: pd.DataFrame, scaler: int):
    if scaler == 0:
        scaler_model = StandardScaler()
    elif scaler == 1:
        scaler_model = MinMaxScaler(feature_range=(0, 1))
    else:
        scaler_model = MinMaxScaler(feature_range=(-1, 1))

    mapper = DataFrameMapper([(data.columns, scaler_model)])
    scaled_features = mapper.fit_transform(data.copy())
    scaled_features_df = pd.DataFrame(
        scaled_features, index=data.index, columns=data.columns
    )

    return scaled_features_df


def treat_categorical_data(data: pd.DataFrame, encoder):
    if encoder == 0:
        labeled_features_df = pd.get_dummies(data=data, prefix=data.columns)
    else:
        le = LabelEncoder()
        data2 = data.copy()
        for i in data.columns:
            data2[i] = le.fit_transform(data2[i])
        labeled_features_df = data2

    return labeled_features_df


def resample_data_stratified(data: pd.DataFrame, column: str):
    min_sample = min(data[column].value_counts())
    resample_data = (
        data.groupby(column, group_keys=False)
        .apply(lambda x: x.sample(min_sample))
        .reset_index(drop=True)
    )
    return resample_data


def inputation_categorical_data(inputation_type: int):
    if inputation_type == 0:
        imp = SimpleImputer(strategy="most_frequent")
    elif inputation_type == 1:
        imp = SimpleImputer(strategy="constant", fill_value="Nao se aplica")
    return imp


def inputation_numerical_data(inputation_type: int):
    if inputation_type == 0:
        imp = SimpleImputer(strategy="mean")
    elif inputation_type == 1:
        imp = SimpleImputer(strategy="median")
    elif inputation_type == 2:
        imp = SimpleImputer(strategy="most_frequent")
    if inputation_type == 3:
        imp = SimpleImputer(strategy="constant", fill_value=0)

    return imp


def remove_outliers(numerical_data: pd.DataFrame, column: str):

    q25, q75 = np.percentile(numerical_data[column], 25), np.percentile(
        numerical_data[column], 75
    )
    iqr = q75 - q25
    lim_inferior = q25 - 1.5 * iqr
    lim_superior = q75 + 1.5 * iqr
    numerical_data = numerical_data.loc[
        (numerical_data[column] > lim_inferior)
        and (numerical_data[column] > lim_superior),
        :,
    ]
    return numerical_data


class preprocessing_initial_data:
    def __init__(
        self,
        remove_outlier=False,
        columns_with_outliers=[],
        resample=False,
        target_column=None,
    ):
        self.remove_outlier = remove_outlier
        self.columns_with_outliers = columns_with_outliers
        self.resample = resample
        self.target_column = target_column

    def fit(self, X):
        pass

    def transform(self, X):
        if self.resample:
            X = resample_data_stratified(X, self.target_column)
        if self.remove_outlier:
            for columns in self.columns_with_outliers:
                X = remove_outliers(X, columns)
        return X


class categorical_tranformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        inputation_categorical=True,
        inputation_type=0,
        treat_categorica=False,
        treat_type=0,
    ):
        inputation_categorical = inputation_categorical
        inputation_type = inputation_type
        treat_categorica = treat_categorica
        treat_type = treat_type

    def fit(self, X):
        return self

    def transform(self, X):
        if self.treat_categorical:
            X = treat_numerical_data(X, self.treat_type)
        if self.inputation_categorical:
            X = inputation_numerical_data(X, self.inputation_type)
        return X


class numerical_tranformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        inputation_numerical=False,
        inputation_type=0,
        treat_numerical=True,
        treat_type=0,
    ):
        self.inputation_numerical = inputation_numerical
        self.inputation_type = inputation_type
        self.treat_numerical = treat_numerical
        self.treat_type = treat_type

    def fit(self, X):
        return self

    def transform(self, X):
        if self.treat_numerical:
            X = treat_numerical_data(X, self.treat_type)
        if self.inputation_numerical:
            X = inputation_numerical_data(X, self.inputation_type)
        return X

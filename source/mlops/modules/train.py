from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin

from source.mlops.modules.preprocessing import *


class Target_encoder(BaseEstimator, TransformerMixin):
    """this class applies the label encoding on a column, it was made with
    intention of applies the encoder on the target column

    Args:
        BaseEstimator (_type_): this class extends the class sklearn.base.BaseEstimator
        TransformerMixin (_type_): this class extends the class sklearn.base.TransformerMixin
    """

    def __init__(self) -> None:
        """object builder

        Returns:
            self: Target_encoder
        """
        return None

    def fit(self, y):
        """this function fits the label encoder function of sklearn

        Args:
            y (list): the target column as a vector
        """
        self.encoder = LabelEncoder()
        self.encoder.fit(y)

    def transform(self, y):
        """this function fit_transform the label encoder for the target column

        Args:
            y (list): the target column as a vector

        Returns:
            _type_: _description_
        """
        return self.encoder.fit_transform(y)


class Model_pipeline(BaseEstimator, TransformerMixin):
    """this class build a pipeline object with the union of preprocessing pipeline and the machine
    learning model, now a days limited as decisionTreeClassifier

    Args:
        BaseEstimator (_type_): this class extends the class sklearn.base.BaseEstimator
        TransformerMixin (_type_): this class extends the class sklearn.base.TransformerMixin
    """

    def __init__(self, model="decisionTreeClassifier"):
        """class builder

        Args:
            model (str, optional):name of the machine learning model the user want to insert into the pipeline.
            Defaults to "decisionTreeClassifier".
        """
        self.model = model

    def fit(self, X, y):
        """create a pipeline with the union of a preprocessing pipeline that comes from souce.mlops.modules.preprocessing
        proper module. and train the full pipeline with the data.

        Args:
            X (pd.DataFrame): full data to train the model
            y (pd.DataFrame): target column as array

        """
        self.model_pipeline = Pipeline(
            steps=[
                ("preprocess", pipeline_preprocessing()),
                (
                    "train",
                    DecisionTreeClassifier(
                        criterion="gini", splitter="best", max_depth=4
                    ),
                ),
            ]
        )
        # splitting the data
        # FIXME: must send the test data for the wandb to test module
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self.model_pipeline.fit(X_train, y_train)
        return self

    def predict(self, X):
        """receives the data and returns it's prevision according to the model trained in the
        fit function

        Args:
            X (pd.DataFrame)): raw_data to predict

        Returns:
            'prediction'(np.array): predict result
        """
        return self.model_pipeline.predict(X)

    def transform(self, X, y):
        """applies the transformation in the data and return's it transformed

        Args:
            X (pd.DataFrame): raw data with the full data without the target columns
            y (pd.DataFrame): raw data of the target column

        Returns:
            pd.DataFrame: transformed data
        """
        self.model_pipeline.transform(X, y)
        return self.model_pipeline

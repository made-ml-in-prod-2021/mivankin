"""
Transformer for scip features in Dataframe
"""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer for scip features in Dataframe
    """
    def __init__(self, features):
        """
        Features transformer constructor
        """
        self.features = features

    def fit(self, X, y=None):
        """
        Features transformer fit function
        """
        return self

    def transform(self, X, y=None):
        """
        Features transformer transform function
        """
        return X[self.features]

    def fit_transfrom(self, X, y=None):
        """
        Features transformer fit_transform function
        """
        return X[self.features]

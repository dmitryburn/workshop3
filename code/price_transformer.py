import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PriceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.non_numeric_values = None

    def fit(self, X, y=None):
        numeric_values = pd.to_numeric(X['Price'], errors='coerce')
        non_numeric_mask = pd.isnull(numeric_values)
        self.non_numeric_values = X['Price'][non_numeric_mask]
        return self

    def transform(self, X, y=None,drop_outlers=True):
        X_transformed = X.copy()

        if self.non_numeric_values is not None:
            X_transformed = X_transformed[~X_transformed.index.isin(self.non_numeric_values.index)]

        X_transformed['Price'] = pd.to_numeric(X_transformed['Price'], errors='coerce')
        if drop_outlers:
            Q1 = X_transformed['Price'].quantile(0.15)
            Q3 = X_transformed['Price'].quantile(0.85)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            X_transformed = X_transformed[(X_transformed['Price'] >= lower_bound) & (X_transformed['Price'] <= upper_bound)]

        X_transformed = X_transformed[~X_transformed.Price.isnull()]
        return X_transformed
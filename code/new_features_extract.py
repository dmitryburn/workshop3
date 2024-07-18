import pandas as pd
import numpy as np
import re
from collections import Counter
import datetime
from sklearn.base import BaseEstimator, TransformerMixin

class TargetFeaturesExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.overall_average_price_cyl = None
        self.price_per_cylinder_dict = {}
        self.brand_avg_price_dict = {}

    def fit(self, X, y=None):
        self.fit_price_per_cylinder(X)
        self.fit_brand_avg_price(X)
        return self

    def transform(self, X):
        X['PricePerCylinder'] = self.transform_price_per_cylinder(X)
        X['MeanPriceBrand'] = self.transform_brand_avg_price(X)
        return X

    def fit_price_per_cylinder(self,X):
        X_transformed = X.copy()
        def convert_cylinders(cyl):
            if '-' in cyl or 'L' in cyl:
                return float(4)
            return float(cyl.replace(' cyl', ''))
        
        X_transformed['Cylinders'] = X_transformed['CylindersinEngine'].apply(convert_cylinders) 
        average_price_per_cylinders = X_transformed.groupby('Cylinders')['Price'].mean()

        self.overall_average_price_cyl = X_transformed['Price'].mean()

        self.price_per_cylinder_dict = average_price_per_cylinders.to_dict()
        

    def transform_price_per_cylinder(self,X):
        X_transformed = X.copy()

        def convert_cylinders(cyl):
            if '-' in cyl or 'L' in cyl:
                return float(4)
            return float(cyl.replace(' cyl', ''))
        
        X_transformed['Cylinders'] = X_transformed['CylindersinEngine'].apply(convert_cylinders) 
        
        def get_average_price(cylinders):
            return  self.price_per_cylinder_dict.get(cylinders, self.overall_average_price_cyl)
        X_transformed['AveragePrice'] = X_transformed['Cylinders'].apply(get_average_price)  

        return  X_transformed.AveragePrice

    def fit_brand_avg_price(self, X):
        self.brand_avg_price_dict = X.groupby('Brand')['Price'].mean().to_dict()

    def transform_brand_avg_price(self, X):
        X_transformed = X.copy()
        def get_brand_avg_price(brand):
            return self.brand_avg_price_dict.get(brand, self.overall_average_price_cyl)
        X_transformed['MeanPriceBrand'] = X_transformed['Brand'].apply(get_brand_avg_price)
        return X_transformed['MeanPriceBrand']

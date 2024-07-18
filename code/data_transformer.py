import pandas as pd
import numpy as np
import re
from collections import Counter
import datetime
from sklearn.base import BaseEstimator, TransformerMixin

class BaseDataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold_brands=10, threshold_cylindersin=86, threshold_models=100, num_new_features=15,another_new_features=False):
        self.num_new_features = num_new_features
        self.threshold_brands = threshold_brands
        self.threshold_cylindersin = threshold_cylindersin
        self.threshold_models = threshold_models

        self.rare_categories_brand = None
        self.rare_categories_cylindersin = None
        self.rare_categories_model = None

        self.non_numeric_values = None

        self.moda_bodytype = None
        self.moda_doors = None
        self.moda_seats = None
        self.moda_location = None
        self.median_kilometers = None
        self.literes_median = {'Electric': None, 'Hybrid': None, 'Other': None}
        self.consumption_literes_median = {'Electric': None, 'Hybrid': None, 'Other': None}
        self.price_per_cylinder_dict = {}
        self.new_features_list = []
        self.max_year = None
        self.another_new_features = another_new_features


    def fit(self, X, y=None):
        self.max_year = X.Year.max()
        self.moda_bodytype = X.BodyType.mode()[0]
        self.moda_doors = X.Doors.mode()[0]
        self.moda_seats = X.Seats.mode()[0]
        self.moda_location = X.Location.mode()[0]
        self.median_kilometers = X[(X.Kilometres != '-') & (X.Kilometres != '- / -')].Kilometres.astype(int).median()

        self.rare_categories_model = self.fit_rare_categories(X.Model, self.threshold_models)
        self.rare_categories_brand = self.fit_rare_categories(X.Brand, self.threshold_brands)
        self.rare_categories_cylindersin = self.fit_rare_categories(X.CylindersinEngine, self.threshold_cylindersin)
        self.fit_literes_median(X)
        self.fit_consumption_literes_median(X)
        self.new_features_list = self.fit_new_features(X)
        return self

    def transform(self, X):
        X_transformed = X.copy()


        X_transformed.Brand = X_transformed.Brand.apply(lambda x: 'Other' if x in self.rare_categories_brand else x)

        X_transformed.Model = X_transformed.Model.apply(lambda x: 'Other' if x in self.rare_categories_model else x)

        X_transformed.FuelConsumption = self.transform_consumption_literes(X_transformed)

        X_transformed.CylindersinEngine = X_transformed.CylindersinEngine.apply(lambda x: '-' if x in self.rare_categories_cylindersin else x)

        X_transformed.Engine = self.transform_literes(X_transformed)

        X_transformed.BodyType = self.transform_BodyType(X_transformed)

        X_transformed.FuelType = self.transform_FuelType(X_transformed)

        X_transformed.Transmission = self.transform_Transmission(X_transformed)

        X_transformed.Kilometres = self.transform_Kilometres(X_transformed)

        X_transformed.ColourExtInt = self.transform_ColourExtInt(X_transformed)

        X_transformed = self.transform_new_features(X_transformed)

        X_transformed.Location = self.transform_location(X_transformed)

        X_transformed.Doors = self.transform_Doors(X_transformed)

        X_transformed.Seats = self.transform_Seats(X_transformed)
        
        X_transformed = self.add_new_features(X_transformed)
         
        if self.another_new_features:
            X_transformed['ColorLocation'] = X_transformed['ColourExtInt'] + '_' + X_transformed['Location']
            X_transformed['FuelAndTransmission'] = X_transformed['FuelType'] + '_' + X_transformed['Transmission']
            X_transformed['DriveBody'] = X_transformed['DriveType'] + '_' + X_transformed['BodyType']
            X_transformed['HightQualityPower'] = ((X_transformed['Engine'] > 3.0) & (X_transformed['Transmission'] == 'Automatic')).astype(int).astype(str)
            X_transformed['LastModel'] = (X_transformed['Year'] == self.max_year).astype(int).astype(str)
            X_transformed['TrendFuelConsumption'] = X_transformed.groupby(['Brand', 'Model'])['FuelConsumption'].transform(lambda x: x.diff().mean())

        return X_transformed.drop(['Car/Suv','Title'], axis=1)
    
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



    def add_new_features(self,X):
        X_transformed = X.copy()

        current_year = datetime.datetime.now().year
        car_age = current_year - X_transformed['Year']

        X_transformed['AvgKilometresPerYear'] = X_transformed['Kilometres'] / car_age

        def convert_cylinders(cyl):
            if '-' in cyl or 'L' in cyl:
                return float(4)
            return float(cyl.replace(' cyl', ''))

        cylsin = X_transformed['CylindersinEngine'].apply(convert_cylinders)


        X_transformed['EnginePerCylinder'] = X_transformed['Engine'] / cylsin

        #X_transformed['FuelConsumptionPerCylinder'] = X_transformed['FuelConsumption'] / cylsin

        return X_transformed

    def transform_Seats(self, X):
        X_transformed = X.copy()

        def categorize_seats(seats):
            if pd.notna(seats) and 'Seats' in seats:
                num_seats = int(seats.split(' Seats')[0])
                if num_seats >= 9:
                    return '9+ Seats'
                else:
                    return seats
            else:
                return seats

        X_transformed.Seats = X_transformed.Seats.apply(categorize_seats)

        return X_transformed.Seats.fillna(self.moda_seats)



    def transform_Doors(self, X):
        X_transformed = X.copy()
        X_transformed.Doors = X_transformed.Doors.apply(lambda x: np.nan if isinstance(x, str) and 'Seats' in x else x)
        return X_transformed.Doors.fillna(self.moda_doors)

    def transform_ColourExtInt(self, X):
        X_transformed = X.copy()
        X_transformed.ColourExtInt = X_transformed.ColourExtInt.apply(lambda x: x.split(' / ')[0])
        return X_transformed.ColourExtInt.apply(lambda x: 'years' if 'year' in x else x)

    def transform_consumption_literes(self, X):
        X_transformed = X.copy()

        def get_literes(name):
            if name == 'Electric':
                return self.consumption_literes_median['Electric']
            elif name == 'Hybrid':
                return self.consumption_literes_median['Hybrid']
            else:
                return self.consumption_literes_median['Other']

        X_transformed.FuelConsumption = X_transformed.FuelConsumption.apply(lambda x: float(x.split('L')[0]) if isinstance(x, str) and x != '-' else np.nan)

        literes = X_transformed.FuelType.apply(get_literes)
        X_transformed.loc[(X_transformed.FuelConsumption.isna()) | (X_transformed.FuelConsumption == '-'), 'FuelConsumption'] = literes[(X_transformed.FuelConsumption.isna()) | (X_transformed.FuelConsumption == '-')]

        return X_transformed.FuelConsumption.astype(float)


    def transform_literes(self, X):
        X_transformed = X.copy()

        def get_literes(name):
            if name == 'Electric':
                return self.literes_median['Electric']
            elif name == 'Hybrid':
                return self.literes_median['Hybrid']
            else:
                return self.literes_median['Other']

        def extract_liters(value):
            match = re.search(r'(\d+\.?\d*) L', value)
            return float(match.group(1)) if match else np.nan

        X_transformed.Engine = X_transformed.Engine.apply(extract_liters)

        literes = X_transformed.FuelType.apply(get_literes)
        X_transformed.loc[(X_transformed.Engine.isna()) | (X_transformed.Engine == '-'), 'Engine'] = literes[(X_transformed.Engine.isna()) | (X_transformed.Engine == '-')]

        return X_transformed.Engine.astype(float)

    def fit_consumption_literes_median(self, X):
        X_transformed = X.copy()

        X_transformed.FuelConsumption = X_transformed.FuelConsumption.apply(lambda x: float(x.split('L')[0]) if isinstance(x, str) and x != '-' else np.nan)

        electric_median = X_transformed[X_transformed.FuelType == 'Electric']['FuelConsumption'].median()
        hybrid_median = X_transformed[X_transformed.FuelType == 'Hybrid']['FuelConsumption'].median()
        other_median = X_transformed[~X_transformed.FuelType.isin(['Electric', 'Hybrid'])]['FuelConsumption'].median()

        self.consumption_literes_median['Electric'] = electric_median
        self.consumption_literes_median['Hybrid'] = hybrid_median
        self.consumption_literes_median['Other'] = other_median

    def fit_literes_median(self, X):
        X_transformed = X.copy()

        def extract_liters(value):
            match = re.search(r'(\d+\.?\d*) L', value)
            return float(match.group(1)) if match else np.nan

        X_transformed.Engine = X_transformed.Engine.apply(extract_liters)

        electric_median = X_transformed[X_transformed.FuelType == 'Electric']['Engine'].median()
        hybrid_median = X_transformed[X_transformed.FuelType == 'Hybrid']['Engine'].median()
        other_median = X_transformed[~X_transformed.FuelType.isin(['Electric', 'Hybrid'])]['Engine'].median()

        self.literes_median['Electric'] = electric_median
        self.literes_median['Hybrid'] = hybrid_median
        self.literes_median['Other'] = other_median

    def transform_new_features(self, X):
        X_transformed = X.copy()
        for value in self.new_features_list:
            X_transformed[value] = X_transformed.Title.apply(lambda x: '1' if value in x else '0')
        return X_transformed

    def fit_new_features(self,X):
        X_transformed = X.copy()
        def remove_first_three_words(text):
            return ' '.join(text.split()[3:]) if len(text.split()) > 3 else ''

        def remove_words_in_parentheses(text):
            text = re.sub(r'\([^)]*\)', '', text)
            text = re.sub(r'\b\d+\.\d+\b', '', text)
            text = re.sub(r'\b\d+', '', text)
            return text


        temp = X_transformed.Title.apply(remove_first_three_words)
        temp = temp.apply(remove_words_in_parentheses)

        all_words = ' '.join(temp).split()

        word_counts = Counter(all_words)

        top_words = word_counts.most_common(self.num_new_features)

        return list(map(lambda x: x[0],top_words))

    def transform_location(self, X):
        X_transformed = X.copy()
        X_transformed.loc[X_transformed.Location != self.moda_location, 'Location'] = 'Other'
        return X_transformed.Location

    def transform_BodyType(self, X):
        X_transformed = X.copy()
        X_transformed.loc[X_transformed.BodyType != self.moda_bodytype, 'BodyType'] = 'Other'
        return X_transformed.BodyType
        
    def transform_FuelType(self,X):
        X_transformed = X.copy()
        def fill_cat(value):
            if value == 'LPG':
                return 'Electric'
            elif value == 'Leaded':
                return 'Premium'
            else:
                return value
        return X_transformed.FuelType.apply(fill_cat)

    def transform_Transmission(self,X):
        X_transformed = X.copy()
        return X_transformed.Transmission.apply(lambda x: 'Automatic' if x == '-' else x)


    def transform_Kilometres(self, X):
        X_transformed = X.copy()
        X_transformed.loc[X_transformed.Kilometres == '-', 'Kilometres'] = self.median_kilometers
        X_transformed.loc[X_transformed.Kilometres == '- / -', 'Kilometres'] = self.median_kilometers
        X_transformed.Kilometres = X_transformed.Kilometres.astype(float)
        return X_transformed.Kilometres

    def fit_rare_categories(self, feature, threshold):
        counts = feature.value_counts()
        return counts[counts <= threshold].index.tolist()
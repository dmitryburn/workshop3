import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor, Pool
from sklearn.preprocessing import OneHotEncoder

class GreedyFeatureSelection:
    def __init__(self,iterations=7000, verbose=1000):
        self.model = CatBoostRegressor(task_type="CPU", iterations=iterations, verbose=verbose)
        self.chosen_numeric_cols = []
        self.chosen_categorical_cols = []
        self.logs = {}

    def __preprocess_data(self, X_train, X_test, numeric_columns,categorical_columns,one_hot_encoding=False):
        if one_hot_encoding:
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(),numeric_columns),
                    ('cat', OneHotEncoder(handle_unknown='ignore'),categorical_columns),  
            ],
            remainder='passthrough' 
        )
        else:
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numeric_columns),
                ],
                remainder='passthrough'
            )            
        
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
        return X_train_transformed, X_test_transformed

    def feature_selection(self, X_train, y_train, X_test, y_test, numeric_columns, categorical_columns,one_hot_encoding=False,improvement_threshold=0.005):

        new_features_categorical = list(X_train.drop(numeric_columns,axis=1).columns)
        X_train, X_test = self.__preprocess_data(X_train, X_test, numeric_columns,new_features_categorical,one_hot_encoding=one_hot_encoding)
        
        full_cols_names = numeric_columns + new_features_categorical
        X_train = pd.DataFrame(X_train,columns=full_cols_names)
        X_test = pd.DataFrame(X_test,columns=full_cols_names)

        res_cols = []
        best_score = -1
        print(f'Количество признаков = {X_train.shape[1]}')
        
        for col in full_cols_names:
            print('--------------------------------------------------')
            print(f'Изучаем колонку {col}')
            temp_cols = res_cols + [col]
            print(f'Сейчас используются колонки с именами: {temp_cols}')
            
            if col in categorical_columns:
                cat_features = self.chosen_categorical_cols + [col]
            else:
                cat_features = self.chosen_categorical_cols

            train_pool = Pool(data=X_train[temp_cols],
                              label=y_train,
                              cat_features=cat_features)
            test_pool = Pool(data=X_test[temp_cols], cat_features=cat_features)

            self.model.fit(train_pool)

            y_pred = self.model.predict(test_pool)

            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = mse ** 0.5
            r2 = r2_score(y_test, y_pred)

            print(f"MAE: {mae:.5f}")
            print(f"RMSE: {rmse:.5f}")
            print(f"R2: {r2:.5f}")

            if r2 > best_score + improvement_threshold:
                best_score = r2
                res_cols.append(col)
                if col in categorical_columns:
                    self.chosen_categorical_cols.append(col)
                else:
                    self.chosen_numeric_cols.append(col)

        self.logs['best_features'] = res_cols
        self.logs['best_categorical_cols'] = self.chosen_categorical_cols
        self.logs['best_numeric_cols'] = self.chosen_numeric_cols
        self.logs['best_r2_score'] = best_score

        return res_cols, self.chosen_categorical_cols




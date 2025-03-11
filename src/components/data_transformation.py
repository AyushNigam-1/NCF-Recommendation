import sys
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from src.exception.exception import CustomException  # Assuming you have this
from src.logging.logger import logging  # Assuming you have this
from src.utils.main_utils.utils import save_numpy_array_data,save_object
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact
)
from src.entity.config_entity import DataTransformationConfig


# @dataclass
# class DataTransformationConfig:
#     transformed_object_file_path: str = os.path.join("artifacts", "preprocessor.pkl")
#     transformed_train_file_path: str = os.path.join("artifacts", "transformed_train.npy")
#     transformed_test_file_path: str = os.path.join("artifacts", "transformed_test.npy")
#     label_encoders_file_path: str = os.path.join("artifacts", "label_encoders.pkl")

# @dataclass
# class DataTransformationArtifact:
#     transformed_object_file_path: str
#     transformed_train_file_path: str
#     transformed_test_file_path: str
#     label_encoders_file_path: str

class DataTransformation:
    def __init__(self,data_validation_artifact:DataValidationArtifact,
                data_transformation_config:DataTransformationConfig):
        try:
            self.data_validation_artifact:DataValidationArtifact=data_validation_artifact
            self.data_transformation_config:DataTransformationConfig=data_transformation_config
        except Exception as e:
            raise CustomException(e,sys)
        
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys)

    def get_data_transformer_object(self, categorical_features, numeric_cols, ordinal_categories):
        try:
            logging.info("Creating data transformer object...")
            
            # Define the ordinal encoder with the given category order
            ordinal_encoder = OrdinalEncoder(
                categories=[ordinal_categories[col] for col in categorical_features],
                dtype=int,
                handle_unknown='use_encoded_value',
                unknown_value=-1,  # Fix: Use a float value instead of np.nan
                # dtype=np.float64    # Fix: Ensure dtype is float to match NaNs
            )
            # Standard Scaler for numerical features
            scaler = StandardScaler()
            
            # Define transformations
            cat_transformers = [("ordinal", ordinal_encoder, categorical_features)]
            num_transformers = [("scaler", scaler, numeric_cols)]
            
            # Combine transformers using ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=cat_transformers + num_transformers,
                remainder='passthrough'
            )
            
            logging.info("Data transformer object created successfully.")
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def perform_feature_engineering(self, df):
        # All of your complex feature engineering logic goes here
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
        df= df[(df['Quantity'] >= 0) & (df['UnitPrice'] >= 0)]
        df = df.copy()  # Ensure df is not a slice

        df.loc[:, 'hour'] = df['InvoiceDate'].dt.hour
        df.loc[:, 'weekday'] = df['InvoiceDate'].dt.weekday
        df.loc[:, 'month'] = df['InvoiceDate'].dt.month
        df.loc[:, 'week'] = df['InvoiceDate'].dt.isocalendar().week
        df.loc[:, 'total_sales'] = df['UnitPrice'] * df['Quantity']

        def calculate_peak_periods(data, group_by_column, top_n_percent=20):
            period_sales = data.groupby(group_by_column)['total_sales'].sum().reset_index()
            period_sales_sorted = period_sales.sort_values(by='total_sales', ascending=False)
            num_peak_periods = int((top_n_percent / 100) * len(period_sales_sorted))
            peak_periods = period_sales_sorted.head(num_peak_periods)[group_by_column].tolist()
            return peak_periods

        peak_months = calculate_peak_periods(df, 'month')
        peak_weeks = calculate_peak_periods(df, 'week')
        peak_weekdays = calculate_peak_periods(df, 'weekday')
        peak_hours = calculate_peak_periods(df, 'hour')
        print("Peak Months:", peak_months)
        print("Peak Weeks:", peak_weeks)
        print("Peak Weekdays:", peak_weekdays)
        print("Peak Hours:", peak_hours)

        def assign_peak_level_combined(row):
            peak_count = sum([
                row['month'] in peak_months,
                row['week'] in peak_weeks,
                row['weekday'] in peak_weekdays,
                row['hour'] in peak_hours
            ])
            return ['Very Low', 'Low', 'Medium', 'High', 'Very High'][peak_count]

        df.loc[:, 'peak_period_level'] = df.apply(assign_peak_level_combined, axis=1)
        product_demand = df.groupby('StockCode')['Quantity'].sum().reset_index()
        product_demand.rename(columns={'Quantity': 'total_quantity'}, inplace=True)
        high_demand_threshold = np.percentile(product_demand['total_quantity'], 80)
        low_demand_threshold = np.percentile(product_demand['total_quantity'], 20)

        def assign_demand_level(total_quantity):
            if total_quantity > high_demand_threshold:
                return 'High'
            elif total_quantity < low_demand_threshold:
                return 'Low'
            else:
                return 'Medium'

        product_demand['overall_demand_level'] = product_demand['total_quantity'].apply(assign_demand_level)
        df = df.merge(product_demand[['StockCode', 'overall_demand_level']], on='StockCode')
        reference_date = df['InvoiceDate'].max()
       
        print(reference_date)
        rfm = df.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (reference_date - x.max()).days,
            'InvoiceNo': 'count',
            'UnitPrice': 'sum'
        }).reset_index()
        rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
        # rfm['RecencySegment'] = pd.qcut(rfm['Recency'], 4, labels=['High', 'Medium', 'Low', 'Very Low'], duplicates='drop')
        # rfm['FrequencySegment'] = pd.qcut(rfm['Frequency'], 4, labels=['Very Low', 'Low', 'Medium', 'High'], duplicates='drop')
        # rfm['MonetarySegment'] = pd.qcut(rfm['Monetary'], 4, labels=['Very Low', 'Low', 'Medium', 'High'], duplicates='drop')
        num_bins = 4
        recency_bins = pd.qcut(rfm['Recency'], num_bins, duplicates='drop')
        frequency_bins = pd.qcut(rfm['Frequency'], num_bins, duplicates='drop')
        monetary_bins = pd.qcut(rfm['Monetary'], num_bins, duplicates='drop')
        rfm['RecencySegment'] = pd.qcut(rfm['Recency'], num_bins, labels=[f"Q{i+1}" for i in range(len(recency_bins.cat.categories))], duplicates='drop')
        rfm['FrequencySegment'] = pd.qcut(rfm['Frequency'], num_bins, labels=[f"Q{i+1}" for i in range(len(frequency_bins.cat.categories))], duplicates='drop')
        rfm['MonetarySegment'] = pd.qcut(rfm['Monetary'], num_bins, labels=[f"Q{i+1}" for i in range(len(monetary_bins.cat.categories))], duplicates='drop')
        df = df.merge(rfm[['CustomerID', 'RecencySegment','FrequencySegment','MonetarySegment']], on='CustomerID', how='left')
        all_countries_stats = df.groupby('Country').agg(
            total_sales=('total_sales', 'sum'),
            avg_unit_price=('UnitPrice', 'mean')
        ).reset_index()
        countries_excluding_uk_stats = all_countries_stats[all_countries_stats['Country'] != 'United Kingdom']
        high_demand_threshold_excluding_uk = np.percentile(countries_excluding_uk_stats['total_sales'], 80)
        low_demand_threshold_excluding_uk = np.percentile(countries_excluding_uk_stats['total_sales'], 20)
        high_demand_threshold_including_uk = np.percentile(all_countries_stats['total_sales'], 80)
        low_demand_threshold_including_uk = np.percentile(all_countries_stats['total_sales'], 20)

        def demand_level(country, total_sales):
            high_threshold, low_threshold = (
                (high_demand_threshold_including_uk, low_demand_threshold_including_uk)
                if country == 'United Kingdom'
                else (high_demand_threshold_excluding_uk, low_demand_threshold_excluding_uk)
            )
            return 'High' if total_sales > high_threshold else 'Low' if total_sales < low_threshold else 'Medium'
        all_countries_stats['country_purchasing_power'] = all_countries_stats.apply(
            lambda x: demand_level(x['Country'], x['total_sales']), axis=1
        )
        df = df.merge(all_countries_stats[['Country', 'country_purchasing_power']], on='Country', how='left')
        grouped_sales = df.groupby(['StockCode', 'Country'])['total_sales'].sum().reset_index()

        def calculate_thresholds(group):
            thresholds = {
                'very_low_threshold': group['total_sales'].quantile(0.10),
                'low_threshold': group['total_sales'].quantile(0.25),
                'medium_threshold': group['total_sales'].quantile(0.50),
                'high_threshold': group['total_sales'].quantile(0.75),
                'very_high_threshold': group['total_sales'].quantile(0.90),
            }
            return pd.Series(thresholds)

        thresholds = grouped_sales.groupby('StockCode').apply(calculate_thresholds).reset_index()
        grouped_sales = grouped_sales.merge(thresholds, on='StockCode')

        def assign_level(row):
            if row['total_sales'] <= row['very_low_threshold']:
                return 'Very Low'
            elif row['total_sales'] <= row['low_threshold']:
                return 'Low'
            elif row['total_sales'] <= row['medium_threshold']:
                return 'Medium'
            elif row['total_sales'] <= row['high_threshold']:
                return 'High'
            else:
                return 'Very High'

        grouped_sales['sales_level_by_country'] = grouped_sales.apply(assign_level, axis=1)
        df = df.merge(grouped_sales[['StockCode', 'Country', 'sales_level_by_country']],on=['StockCode','Country'],how='left')

        def adjust_unit_price(row):
            price_adjustment = 0

            if row['peak_period_level'] == 'High':
                price_adjustment += 0.10
            elif row['peak_period_level'] == 'Very High':
                price_adjustment += 0.20
            elif row['peak_period_level'] == 'Low':
                price_adjustment -= 0.05
            elif row['peak_period_level'] == 'Very Low':
                price_adjustment -= 0.10

            if row['overall_demand_level'] == 'High':
                price_adjustment += 0.15
            elif row['overall_demand_level'] == 'Medium':
                price_adjustment += 0.05
            elif row['overall_demand_level'] == 'Low':
                price_adjustment -= 0.10

            if row['RecencySegment'] == 'High':
                price_adjustment += 0.05
            elif row['RecencySegment'] == 'Medium':
                price_adjustment += 0.03
            elif row['RecencySegment'] == 'Low' or row['RecencySegment'] == 'Very Low':
                price_adjustment -= 0.05

            if row['FrequencySegment'] == 'High':
                price_adjustment += 0.10
            elif row['FrequencySegment'] == 'Medium':
                price_adjustment += 0.05
            elif row['FrequencySegment'] == 'Low' or row['FrequencySegment'] == 'Very Low':
                price_adjustment -= 0.05

            if row['MonetarySegment'] == 'High':
                price_adjustment += 0.10
            elif row['MonetarySegment'] == 'Medium':
                price_adjustment += 0.05
            elif row['MonetarySegment'] == 'Low' or row['MonetarySegment'] == 'Very Low':
                price_adjustment -= 0.05

            if row['country_purchasing_power'] == 'High':
                price_adjustment += 0.08
            elif row['country_purchasing_power'] == 'Medium':
                price_adjustment += 0.04
            elif row['country_purchasing_power'] == 'Low':
                price_adjustment -= 0.08

            if row['sales_level_by_country'] == 'Very High':
                price_adjustment += 0.15
            elif row['sales_level_by_country'] == 'High':
                price_adjustment += 0.12
            elif row['sales_level_by_country'] == 'Medium':
                price_adjustment += 0.05
            elif row['sales_level_by_country'] == 'Low':
                price_adjustment -= 0.10
            elif row['sales_level_by_country'] == 'Very Low':
                price_adjustment -= 0.15

            new_unit_price = row['UnitPrice'] * (1 + price_adjustment)
            return new_unit_price

        df['adjusted_unit_price'] = df.apply(adjust_unit_price, axis=1)

        return df
    
    def split_train_test(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
        logging.info("Splitting dataset into train and test sets...")
        
        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        train_df = pd.concat([train_X, train_y], axis=1)
        test_df = pd.concat([test_X, test_y], axis=1)
        
        logging.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
        
        return train_df, test_df

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info("Entered initiate_data_transformation method of DataTransformation class")
        try:
            logging.info("Starting data transformation")
            df = self.read_data(self.data_validation_artifact.valid_data_file_path)
            df = self.perform_feature_engineering(df)

            categorical_features = ['peak_period_level', 'overall_demand_level', 'RecencySegment',
                                    'FrequencySegment', 'MonetarySegment', 'country_purchasing_power', 'sales_level_by_country']

            numeric_cols = ['Quantity', 'UnitPrice', 'total_sales']
            target_column = 'adjusted_unit_price'
            X = df.drop(columns=[target_column])  # Features (all columns except target)
            y = df[target_column] 
            train_df, test_df = self.split_train_test(X, y)
            print("Train columns:", train_df.columns)
            print("Test columns:", test_df.columns)
            print("Expected categorical features:", categorical_features)
            print("Expected numeric features:", numeric_cols)

            ordinal_categories = {
                "peak_period_level": ["Very Low", "Low", "Medium", "High", "Very High"],
                "overall_demand_level": ["Low", "Medium", "High"],
                "RecencySegment": ["Very Low", "Low", "Medium", "High"],
                "FrequencySegment": ["Very Low", "Low", "Medium", "High"],
                "MonetarySegment": ["Very Low", "Low", "Medium", "High"],
                "country_purchasing_power": ["Low", "Medium", "High"],
                "sales_level_by_country": ["Very Low", "Low", "Medium", "High", "Very High"]
            }

            transformer = self.get_data_transformer_object(
                categorical_features=list(ordinal_categories.keys()),
                numeric_cols=['Quantity', 'UnitPrice', 'total_sales'],
                ordinal_categories=ordinal_categories
            )

            input_feature_train_df = train_df[categorical_features + ['UnitPrice', 'Quantity','total_sales']]
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df[categorical_features + ['UnitPrice', 'Quantity','total_sales']]
            target_feature_test_df = test_df[target_column]

            logging.info("Applying preprocessing object to datasets...")
            input_feature_train_arr = transformer.fit_transform(input_feature_train_df)
            input_feature_test_arr = transformer.transform(input_feature_test_df)

            

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path, transformer)
            save_object( "final_model/preprocessor.pkl", transformer)

            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )
            return data_transformation_artifact

        except Exception as e:
            raise CustomException(e, sys)
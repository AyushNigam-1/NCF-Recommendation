import sys
import pandas as pd
from src.exception.exception import CustomException
from src.utils.main_utils.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = "final_model/model.pkl"
            preprocessor_path = "final_model/preprocessor.pkl"
            
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(
        self,
        InvoiceNo: int,
        StockCode: str,
        Description: str,
        Quantity: int,
        InvoiceDate: str,
        UnitPrice: float,
        CustomerID: float,
        Country: str,
        hour: int,
        weekday: int,
        week: int,
        total_sales: float,
        peak_period_level: str,
        overall_demand_level: str,
        RecencySegment: str,
        FrequencySegment: str,
        MonetarySegment: str,
        country_purchasing_power: str,
        sales_level_by_country: str,
        adjusted_unit_price: float,
    ):
        self.InvoiceNo = InvoiceNo
        self.StockCode = StockCode
        self.Description = Description
        self.Quantity = Quantity
        self.InvoiceDate = InvoiceDate
        self.UnitPrice = UnitPrice
        self.CustomerID = CustomerID
        self.Country = Country
        self.hour = hour
        self.weekday = weekday
        self.week = week
        self.total_sales = total_sales
        self.peak_period_level = peak_period_level
        self.overall_demand_level = overall_demand_level
        self.RecencySegment = RecencySegment
        self.FrequencySegment = FrequencySegment
        self.MonetarySegment = MonetarySegment
        self.country_purchasing_power = country_purchasing_power
        self.sales_level_by_country = sales_level_by_country
        self.adjusted_unit_price = adjusted_unit_price
    
    def get_data_as_data_frame(self):
        try:
            data_dict = {
                "InvoiceNo": [self.InvoiceNo],
                "StockCode": [self.StockCode],
                "Description": [self.Description],
                "Quantity": [self.Quantity],
                "InvoiceDate": [self.InvoiceDate],
                "UnitPrice": [self.UnitPrice],
                "CustomerID": [self.CustomerID],
                "Country": [self.Country],
                "hour": [self.hour],
                "weekday": [self.weekday],
                "week": [self.week],
                "total_sales": [self.total_sales],
                "peak_period_level": [self.peak_period_level],
                "overall_demand_level": [self.overall_demand_level],
                "RecencySegment": [self.RecencySegment],
                "FrequencySegment": [self.FrequencySegment],
                "MonetarySegment": [self.MonetarySegment],
                "country_purchasing_power": [self.country_purchasing_power],
                "sales_level_by_country": [self.sales_level_by_country],
                "adjusted_unit_price": [self.adjusted_unit_price],
            }
            return pd.DataFrame(data_dict)
        except Exception as e:
            raise CustomException(e, sys)

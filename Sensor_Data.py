import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from SensorSync_FrontPage import *

# function for checking the null values
def cleaning(df: pd.DataFrame, col:str) -> str:
    if data_csv is not None:
        print(f"Print the no of null values: {data_csv.isna().sum()}")
        print("=======================================================================")
        print()
        print(f" {data_csv[data_csv['age'].isna()]}")
        print()
        print("========================================================================")
        print(f"fill the null values using forward fill: {data_csv['age'].ffill(inplace=True)}")
        print("========================================================================")
        print()
        print(f"fill the null values using backward fill: {data_csv['age'].bfill(inplace=True)}")
        print()
        print("=======================================================================")
        print(f"Print the no of null values: {data_csv.isna().sum()}")
        print()

    elif data_excel is not None:
        print(f"Print the no of null values: {data_excel.isna().sum()}")
        print("=======================================================================")
        print()
        print(f" {data_excel[data_excel['age'].isna()]}")
        print()
        print("========================================================================")
        print(f"fill the null values using forward fill: {data_excel['age'].ffill(inplace=True)}")
        print("========================================================================")
        print()
        print(f"fill the null values using backward fill: {data_excel['age'].bfill(inplace=True)}")
        print()
        print("=======================================================================")
        print(f"Print the no of null values: {data_excel.isna().sum()}")
        print()




if data_csv is not None:
    combine_copy = data_csv.copy()
    combine_copy['age'].fillna(combine_copy['age'].median(),inplace=True)

elif data_excel is not None:
    combine_copy = data_excel.copy()
    combine_copy['age'].fillna(combine_copy['age'].median(),inplace=True)

# manipulate the date time column
def manipulate_date(df: pd.DataFrame):
    combine_copy['datetime']= pd.to_datetime(combine_copy['datetime'])

# separating the the datetime column into year, month, date and hour
    combine_copy['year'] = combine_copy['datetime'].dt.year
    combine_copy['month'] = combine_copy['datetime'].dt.month
    combine_copy['date'] = combine_copy['datetime'].dt.day
    combine_copy['hour'] = combine_copy['datetime'].dt.hour

# using dictionary
    month = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
            7:'July', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}

    combine_copy['month'] = combine_copy['month'].map(month)

# change the age column to int type
    combine_copy['age'] = combine_copy['age'].astype(int)


def round_value(df: pd.DataFrame, col: float) -> float:
    df[col] = round(df[col], 2)

values = ['volt','rotate', 'pressure', 'vibration']
for i in values:
    round_value(combine_copy, i)
     
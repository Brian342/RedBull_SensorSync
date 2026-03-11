import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# function for checking the null values
def cleaning(data_csv, data_excel: pd.DataFrame, col:str) -> str:
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
    df['datetime']= pd.to_datetime(df['datetime'])

# separating the the datetime column into year, month, date and hour
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['date'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour

# using dictionary
    month = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
            7:'July', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}

    df['month'] = df['month'].map(month)

# change the age column to int type
    df['age'] = df['age'].astype(int)


def round_value(df: pd.DataFrame, col: float) -> float:
    df[col] = round(df[col], 2)

    return df


# values = ['volt','rotate', 'pressure', 'vibration']
# for i in values:
#     round_value(df=combine_copy, col=i)
     
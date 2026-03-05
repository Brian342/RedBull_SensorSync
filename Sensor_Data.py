import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from SensorSync_FrontPage import *

# function for checking the null values
def fillnull(df: pd.DataFrame ,col: str) -> str:
    print(f"Print the no of null values: {df.isna().sum()}")
    print("=======================================================================")
    print()
    print(f" {df[df[col].isna()]}")
    print()
    print("========================================================================")
    print(f"fill the null values using forward fill: {df[col].ffill(inplace=True)}")
    print("========================================================================")
    print()
    print(f"fill the null values using backward fill: {df[col].bfill(inplace=True)}")
    print()
    print("=======================================================================")
    print(f"Print the no of null values: {df.isna().sum()}")
    print()

if data_csv is not None:
    fillnull(data_csv)
elif data_excel is not None:
    fillnull(data_excel)
else:
    print("No file uploaded")


if data_csv is not None:
    combine_copy = data_csv.copy()
    combine_copy['age'].fillna(combine_copy['age'].median(),inplace=True)

elif data_excel is not None:
    combine_copy = data_excel.copy()
    combine_copy['age'].fillna(combine_copy['age'].median(),inplace=True)

# manipulate the date time column

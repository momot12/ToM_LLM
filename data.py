import csv
import pandas as pd


file = "master_csv_test_utf8.csv"

df = pd.read_csv(file)
print(df.head())
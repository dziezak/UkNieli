import pandas as pd
df = pd.read_csv("tox21.csv")
pd.set_option('display.max_columns', None)

print("Kolumnt w zrobirze:")
print(df.columns.tolist())

print("\nPierwsze 5 wierzy:")
print(df.head())

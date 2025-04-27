import pandas as pd

df = pd.read_csv('data/combined_datasets.csv', nrows=5, low_memory=False)
print("Columns in CSV:")
print(df.columns.tolist())
print("\nDoes it have a 'url' column?", 'url' in df.columns)

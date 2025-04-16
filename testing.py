import pandas as pd

df = pd.read_csv("data/combined_datasets.csv", low_memory=False)

print("\n🧪 First 5 rows:")
print(df.head())

print("\n🧩 Columns:")
print(df.columns.tolist())

print("\n📊 Source column value counts (before lowercasing):")
print(df['source'].value_counts(dropna=False))

# If possible, print unique labels
if 'label' in df.columns:
    print("\n🔖 Label value counts:")
    print(df['label'].value_counts(dropna=False))

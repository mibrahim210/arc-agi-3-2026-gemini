import pandas as pd
import sys

try:
    df = pd.read_parquet("scratch/kaggle_output/submission.parquet")
    print(f"Parquet Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("Head (First 20 rows):")
    print(df.head(20))
    print("\nSummary Stats:")
    if 'score' in df.columns:
        print(f"Total rows: {len(df)}")
        print(f"Score sum / stats:\n{df['score'].value_counts()}")
    if 'end_of_game' in df.columns:
        print(f"End of game stats:\n{df['end_of_game'].value_counts()}")
except Exception as e:
    print(f"Error reading parquet: {e}")

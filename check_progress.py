import pandas as pd
df = pd.read_csv('results/multimodel/summary.csv')
print('Completed configs:', len(df))
print(df[['model','dataset','chunk_size','top_k','prompt_strategy']].to_string(index=False))

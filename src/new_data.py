import pandas as pd

file_path = './data/train.csv'
df = pd.read_csv(file_path)


sampled_df = df.sample(frac=0.01, random_state=1)  


output_file_path = './data/train1.csv'
sampled_df.to_csv(output_file_path, index=False)

file_path = './data/test.csv'
df = pd.read_csv(file_path)


sampled_df = df.sample(frac=0.05, random_state=1)  


output_file_path = './data/test1.csv'
sampled_df.to_csv(output_file_path, index=False)


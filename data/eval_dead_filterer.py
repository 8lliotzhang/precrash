# super quick helper program. this reads the v0_eval.csv and returns all where P_ISEV == 2.
import pandas as pd

def filter_dead_evaluations(file_path):
    df = pd.read_csv(file_path)
    return df[df['P_ISEV'] == 2]
# write to v0_eval_dead.csv
filtered_df = filter_dead_evaluations('data/v0_eval.csv')
filtered_df.to_csv('data/v0_eval_dead.csv', index=False)

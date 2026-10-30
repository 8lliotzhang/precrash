# this program takes the 2019 dataset, and filters for the following:

# input values
    # "C_MNTH", "C_HOUR", "C_RCFG", "C_WTHR", "C_RSUR", "C_RALN", "C_TRAF",
    # "V_TYPE", "V_YEAR",
    # "P_SEX", "P_AGE", "P_PSN", "P_SAFE"
# output
    # "P_ISEV"

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

train_fraction = 0.80 
# how much of the model we want to be training;
# 1 - (train_fraction) is eval set fraction.

print("Hello, World!")

# load it
orig = pd.read_csv('data/2019_dataset_en.csv', low_memory=False)
print("Loaded:", len(orig))

# only vehicle drivers & passengers; should have a valid output too
orig = orig[orig["P_USER"].isin(["1", "2"])]
orig = orig[orig["P_ISEV"].isin(["1", "2", "3"])]

# select features and target
features = ["C_MNTH", "C_HOUR", "C_RCFG", "C_WTHR", "C_RSUR", "C_RALN", "C_TRAF", 
            "V_TYPE", "V_YEAR",
            "P_SEX", "P_AGE", "P_PSN", "P_SAFE"]
target = "P_ISEV"

# JUST features n target: drop when N/A, remove other cols.
orig = orig.dropna(subset = features + [target]).copy()
orig = orig[features + [target]].copy()
print("After dropping missing values:", len(orig))

# map y 1,2,3 to 0,1,2. also they should be numeros.
orig[target] = pd.to_numeric(orig[target], errors='raise') - 1

# does the split thing.
train, evaluation = train_test_split(
    orig, 
    train_size = train_fraction, 
    random_state = 67, # as 42 is an elegant number, for a more civilized age.
    stratify = orig[target]
)

# and finally we write em.
train.to_csv('data\\v0_train.csv', index=False)
evaluation.to_csv('data\\v0_eval.csv', index=False)

print("All done :3")
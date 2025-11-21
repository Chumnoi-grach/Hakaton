import numpy as np
import pandas as pd
import os

data = pd.read_csv('/kaggle/input/nstu-hach-ai-track-education-case/data.csv')
marking = pd.read_csv('/kaggle/input/nstu-hach-ai-track-education-case/marking.csv')

ds = data.merge(marking, left_on='PK', right_on='ИД', how='left')
ds.drop('ИД', axis=1, inplace=True)

print(ds.info())
print(ds.head())
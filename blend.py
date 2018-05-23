from os import listdir
import pandas as pd

target_col = 'TARGET'

count = 0
blend = None
weight_sum = 0

for filename in listdir('output/pre_sub'):
    if filename=='.gitkeep':
        continue
    df = pd.read_csv('output/pre_sub/'+filename)
    score = float('0.'+filename.split('_')[1].split('.')[0])
    df[target_col] = score*df[target_col]
    weight_sum += score
    if blend is None:
        blend = df
    else:
        blend[target_col] = blend[target_col] + df[target_col]

blend[target_col] = blend[target_col]/weight_sum
blend.to_csv('blend.csv', index=False)

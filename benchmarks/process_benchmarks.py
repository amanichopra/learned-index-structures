import pandas as pd
import pickle 
import numpy as np

def load_benchmarks(path):
  with open(path, 'rb') as f:
    benchmarks = pickle.load(f)
    benchmarks = pd.DataFrame(benchmarks)
  return benchmarks

def normalize_by_group(df, by, smoothing_constant=0.001):
  groups = df.groupby(by)
  mean = groups.transform("mean")
  std = groups.transform("std")
  normalized = ((df[mean.columns] - mean) / (std + smoothing_constant)).drop('Fold', axis=1)
  df[mean.columns.drop('Fold')] = normalized
  return df
  
def process(benchmarks):
  benchmarks = benchmarks.reset_index().rename(columns={'index': 'Dataset'})

  # process ann benchmarks
  anns = list(benchmarks['ann'][0].keys())
  benchmarks[anns] = float('nan') 
  for ind, row in benchmarks.iterrows():
    for ann in row['ann'].keys():
      benchmarks.loc[ind, ann] = str(row['ann'][ann])
  benchmarks = benchmarks.drop('ann', axis=1)

  benchmarks = pd.melt(benchmarks, id_vars='Dataset', var_name='Model', value_name='Metrics')
  benchmarks[['Predict Time', 'MSE', 'MAE', 'Space']] = benchmarks['Metrics'].apply(pd.Series)
  benchmarks = benchmarks.drop(columns='Metrics')
    
  temp = benchmarks.explode('Predict Time').drop(columns=['MSE', 'MAE', 'Space'])
  temp['MSE'] = benchmarks.explode('MSE')['MSE']
  temp['MAE'] = benchmarks.explode('MAE')['MAE']
  temp['Space'] = benchmarks.explode('Space')['Space']
  temp['Fold'] = [i for i in range(1, 6)] * temp.index.nunique()
  temp = temp.reset_index(drop=True)
  
  return temp
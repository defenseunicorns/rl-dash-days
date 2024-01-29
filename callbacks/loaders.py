import pandas as pd
import numpy as np

def get_metadata():
    return pd.read_csv('./data/metadata.csv')

def get_models():
    md = get_metadata()
    return md.name.unique().tolist()

def get_model_hypers(model):
    md = get_metadata()
    md = md[md['name'] == model]
    md = md.T.reset_index(drop=False).rename(columns={'index':'Param'})
    md.columns = ['Param', 'Value']
    return md

def get_histories(models, last_n):
    frames = []
    if last_n is None:
        last_n = 1
    for model in models:
        try:
            data = pd.read_csv(f'./data/{model}.csv')
        except Exception as e:
            print(f'Error loading model data: {model}, {e}')
            continue
        data['name'] = model
        data = data.reset_index(drop=False).rename(columns={'index':'ts'})
        cutoff = data.ts.max() - last_n
        frames.append(data[data['ts']>cutoff].copy())
    return pd.concat(frames)

def get_model_data(model_name):
    try:
        df = pd.read_csv(f'./data/{model_name}.csv')
    except Exception as e:
        print(f'Error opening model data for model:{model_name}')
        return None
    return df

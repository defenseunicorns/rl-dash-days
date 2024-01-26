import plotly.express as px

import pandas as pd
import numpy as np

from callbacks.loaders import get_histories, get_model_data, get_models

def get_line_graph(model_name):
    df = get_model_data(model_name)
    fig = px.line(df, x=df.index, y=['score', 'actions'],
                 title=f'Training history for {model_name}',
                 labels={'index':'100 Epochs'})
    return fig

def get_box_plot(variable, last_n):
    models = get_models()
    df = get_histories(models, last_n)
    df['score/actions'] = df['score'] / df['actions']
    fig = px.box(df, x='name', y=variable, title=f'Last {last_n} training points',
                labels={'name':'Model'})
    return fig
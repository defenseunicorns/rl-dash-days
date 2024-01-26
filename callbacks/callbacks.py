from dash import dcc, html, Input, Output, State, callback, no_update

from callbacks.loaders import get_models, get_histories, get_model_hypers, get_metadata
from callbacks.plots import get_line_graph, get_box_plot
from test import launch_test

import pandas as pd
import numpy as np
import os

def register_callbacks(app):

    @app.callback(
        Output('model-select', 'options'),
        Output('model-select', 'value'),
        Input('model-select', 'placeholder')
    )
    def get_model_names(ph):
        models = get_models()
        return models, models[0]

    @app.callback(
        Output('model-params', 'data'),
        Input('model-select', 'value')
    )
    def get_model_parameters(model_name):
        params = get_model_hypers(model_name)
        return params.to_dict('records')
    
    @app.callback(
        Output('launch-cmd', 'children'),
        Input('submit-train', 'n_clicks'),
        State('name-input', 'value'),
        State('algo-select', 'value'),
        State('reward-select', 'value'),
        State('epochs-input', 'value'),
        State('death-pen-input', 'value'),
        State('ghost-mult-input', 'value'),
        State('wr-input', 'value'),
        State('wp-input', 'value'),
        State('lambda-r-input', 'value'),
        State('lambda-p-input', 'value')
    )
    def submit_training_request(n_clicks,
                                name,
                                algo,
                                reward_fcn,
                                epochs,
                                death,
                                ghost,
                                wr,
                                wp,
                                lam_r,
                                lam_p):
        if n_clicks:
            if name == "" or algo == "":
                return "Error: Name and Algorithm must be defined"
        
            try:
                cmd = f'python train.py -n {name} -a {algo} '
                if reward_fcn is not None:
                    cmd += f'-r {reward_fcn} '
                if epochs and epochs < 10000 and epochs > 1000:
                    cmd += f'-e {epochs} '
                if death is not None:
                    cmd += f'-d {death} '
                if ghost is not None:
                    cmd += f'-g {ghost} '
                if wr is not None:
                    cmd += f'--reward_weight {wr} '
                if wp is not None:
                    cmd += f'--punish_weight {wp} '
                if lam_r is not None:
                    cmd += f'--reward_memory {lam_r} '
                if lam_p is not None:
                    cmd += f'--punish_memory {lam_p} '
    
                return cmd[:-1]
            except Exception as e:
                return f'Error processing args: {e}'
        else:
            return no_update

    @app.callback(
        Output('trigger_launch', 'children'),
        Input('submit-test', 'n_clicks'),
        Input('model-select', 'value')
    )
    def launch_test_process(n_clicks, model_name):
        if n_clicks:
            md = get_metadata()
            md = md[md['name'] == model_name]
            if len(md) > 0:
                algo = md['algorithm'].values[0]
                launch_test(model_name, algo)

    @app.callback(
        Output('graph', 'figure'),
        Input('model-select', 'value'),
        Input('graph-select', 'value'),
        Input('last-n-input', 'value'),
        Input('box-var-select', 'value')
    )
    def update_graph(model, graph_type, last_n, var):
        if graph_type == 'line':
            return get_line_graph(model)
        else:
            models = get_models()
            return get_box_plot(var, last_n)
    
    @app.callback(
        Output('splitq-div', 'style'),
        Input('algo-select', 'value')
    )
    def change_splitq_visibility(algo):
        if algo == 'Split Q-learn':
            return {'display':'inline'}
        else:
            return {'display':'none'}
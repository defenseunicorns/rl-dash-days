from dash import dcc, html, Input, Output, State, callback, no_update

import pandas as pd
import numpy as np
import os

def register_callbacks(app):
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
                if epochs < 10000 and epochs > 1000:
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
        Output('splitq-div', 'style'),
        Input('algo-select', 'value')
    )
    def change_splitq_visibility(algo):
        if algo == 'Split Q-learn':
            return {'display':'inline'}
        else:
            return {'display':'none'}
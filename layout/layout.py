from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc

import pandas as pd
import numpy as np

header_layout = html.Div([
    html.Div([
        html.Img(src='assets/doug.jpeg')
    ],className='col-sm-auto'),
    html.Div([
        html.H3("Reinforcement Learning Dashboard")
    ], className='col-lg-auto')
], className='row')

train_layout = html.Div([
    html.Div([
        dbc.Card([
            dbc.CardBody([
                html.H4("Training", className="row"),
                html.Div(['Algorithm',
                    dcc.Dropdown(
                        id='algo-select',
                        options=['Deep Q-learn',
                                 'Split Q-learn',
                                 'PPO'
                                ],
                        value='Deep Q-learn')
                ], className='col'),
                html.Div(['Reward function',
                    dcc.Dropdown(
                        id='reward-select',
                        options=[
                            'vanilla',
                            'parameterized',
                            'split_q',
                            'life_penalty',
                            'death_tax',
                            'ghostbusters',
                        ],
                    )
                ], className='col')
            ])
        ])
    ], className='row'),
    html.Div([
        dbc.Card([
            dbc.CardBody([
                html.H4("Shared Parameters", className="row"),
                html.Div([
                    dcc.Input(
                        id='name-select',
                        placeholder='Model Name',
                    ),
                    dcc.Input(id='epochs-input',
                                type='number',
                                placeholder='Epochs',
                                min=1000,
                                max=10000),
                    dcc.Input(id='death-pen-input',
                                type='number',
                                placeholder='Death Penalty',
                                min=50,
                                max=1000),
                    dcc.Input(id='ghost-mult-input',
                              type='number',
                              placeholder='Ghost Multiplier',
                              min=1,
                              max=10),
                ], className='row')
            ])
        ])
    ], className='row'),
    html.Div(id='spliq-div', children=[
        dbc.Card([
            dbc.CardBody([
                html.H4("Split Q-learning parameters", className="row"),
                html.Div([
                    dcc.Input(id='wr-input',
                                type='number',
                                placeholder='Reward Weighting',
                                min=.1,
                                max=1),
                    dcc.Input(id='wp-input',
                                type='number',
                                placeholder='Punishment Weighting',
                                min=.1,
                                max=1),
                    dcc.Input(id='lambda-r-input',
                              type='number',
                              placeholder='Reward memory',
                              min=.1,
                              max=10),
                    dcc.Input(id='lambda-p-input',
                              type='number',
                              placeholder='Punish memory',
                              min=.1,
                              max=10)
                ], className='row')
            ])
        ])
    ], className='row', style={'display':'none'}),
    html.Div([
        dcc.Button(
    ])
], className='col-lg')

app_layout = html.Div([
    header_layout,
    train_layout
])
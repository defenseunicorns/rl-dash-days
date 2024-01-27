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
    ], className='col-lg-auto', style={'margin':'auto', 'text-align':'center'})
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
                                 'Proximal Policy Optimization',
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
                    html.Div([
                        html.Div("Model name", className='col-sm-2'),
                        html.Div([
                            dcc.Input(
                                id='name-input',
                                placeholder='Model Name',
                                style={'width':'100%'},
                            ),
                        ], className='col'),
                    ], className='row'),
                    html.Div([
                        html.Div("Epochs", className='col-sm-2'),
                        html.Div([
                            dcc.Input(
                                id='epochs-input',
                                type='number',
                                placeholder='Epochs',
                                min=1000,
                                max=10000,
                                style={'width':'100%'}
                            ),
                        ], className='col'),
                    ], className='row'),
                    html.Div([
                        html.Div("Death penalty", className='col-sm-2'),
                        html.Div([
                            dcc.Input(
                                id='death-pen-input',
                                type='number',
                                placeholder='Death Penalty',
                                min=50,
                                max=1000,
                                style={'width':'100%'}
                            ),
                        ], className='col'),
                    ], className='row'),
                    html.Div([
                        html.Div("Ghost multiplier", className='col-sm-2'),
                        html.Div([
                        dcc.Input(
                            id='ghost-mult-input',
                            type='number',
                            placeholder='Ghost Multiplier',
                            min=1,
                            max=10,
                            style={'width':'100%'}
                        ),
                        ], className='col'),
                    ], className='row'),
                ], className='row')
            ])
        ])
    ], className='row'),
    html.Div(id='splitq-div', children=[
        dbc.Card([
            dbc.CardBody([
                html.H4("Split Q-learning parameters", className="row"),
                html.Div([
                    html.Div([
                        html.Div("Reward weight", className='col-sm-2'),
                        html.Div([
                            dcc.Input(id='wr-input',
                                        type='number',
                                        placeholder='Reward Weighting',
                                        min=.1,
                                        max=1,
                                        style={'width':'100%'}
                                     ),
                        ], className='col'),
                    ], className='row'),
                    html.Div([
                        html.Div("Punishment weight", className='col-sm-2'),
                        html.Div([
                            dcc.Input(id='wp-input',
                                        type='number',
                                        placeholder='Punishment Weighting',
                                        min=.1,
                                        max=1,
                                        style={'width':'100%'}
                                     ),
                        ],className='col'),
                    ], className='row'),
                    html.Div([
                        html.Div("Reward memory", className='col-sm-2'),
                        html.Div([
                            dcc.Input(id='lambda-r-input',
                                      type='number',
                                      placeholder='Reward memory',
                                      min=.1,
                                      max=10,
                                      style={'width':'100%'}
                                     ),
                        ], className='col'),
                    ], className='row'),
                    html.Div([
                        html.Div("Punishment memory", className='col-sm-2'),
                        html.Div([
                            dcc.Input(id='lambda-p-input',
                                      type='number',
                                      placeholder='Punish memory',
                                      min=.1,
                                      max=10,
                                      style={'width':'100%'}
                                     ),
                        ], className='col'),
                    ], className='row'),
                ], className='row'),
            ]),
        ]),
    ], className='row', style={'display':'none'}),
    html.Div([
        html.Div([
            dbc.Button('Launch training', id='submit-train', n_clicks=0),
        ], className='col'),
    ], className='row'),
    html.Div([], id='launch-cmd'),
], className='col-sm')

eval_layout = html.Div([
    html.Div([
        dbc.Card([
            dbc.CardBody([
                html.H4("Evaluation Options"),
                html.Div([
                    html.Div(["Select a model:", 
                        dcc.Dropdown(
                            id='model-select',
                            options=[
                            ],
                            placeholder='Select a model',
                        ),
                    ], className='col'),
                    html.Div([
                        dbc.Button('Launch Testing', id='submit-test', n_clicks=0),
                    ], className='col'),
                ])
            ]),
        ])
    ], className='row'),
    html.Div([
        dash_table.DataTable(
            id = 'model-params',
            data = None,
            columns=[{"name": i, "id": i} for i in ['Param', 'Value']]
        )
    ], className='row')
], className='col-md')

graph_layout = html.Div([
    dbc.Card([
        dbc.CardBody([
            html.H4("Model Performance"),
            html.Div([
                html.Div(['Graph type',
                    dcc.Dropdown(
                        id='graph-select',
                        options=['line', 'boxplot'],
                        value='line'
                    ),
                ], className='col'),
                html.Div([
                    html.Div(["Truncate to last N epochs"]),
                        dcc.Input(id='last-n-input',
                                  type='number',
                                  value = 10)
                    ], className='col'),
                html.Div(["Variable to plot",
                    dcc.Dropdown(
                        id='box-var-select',
                        options=['score', 'actions', 'score/actions'],
                        value='score',
                    )
                ], className='col'),
            ], className='row'),
            html.Div([
                dcc.Graph(
                    id='graph',
                )
            ], className='row')
        ])
    ])
])

app_layout = html.Div([
    html.Div([
        header_layout,
        train_layout,
        eval_layout
    ], className='row'),
    graph_layout,
    html.Div(id='trigger_launch', children=[]),
], className='row', style={'width':'99%', 'padding-left':'15px'})
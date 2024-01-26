from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc

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
                                ]
                        value='Deep Q-learn'
                    )
                ], className='col-sm'),
                html.Div(['Epochs'
                ], className='col-sm')
            ])
        ])
    ], className='row'),
    html.Div([
        dbc.Card([
            dbc.CardBody([
                html.H4("Hyperparameters", className="row"),
                html.Div([])
            ])
        ])
    ], className='row')
], className='col-lg')

app_layout = html.Div([
    
])

card = dbc.Card(
    [
        dbc.CardImg(src="/static/images/placeholder286x180.png", top=True),
        dbc.CardBody(
            [
                html.H4("Card title", className="card-title"),
                html.P(
                    "Some quick example text to build on the card title and "
                    "make up the bulk of the card's content.",
                    className="card-text",
                ),
                dbc.Button("Go somewhere", color="primary"),
            ]
        ),
    ],
    style={"width": "18rem"},
)
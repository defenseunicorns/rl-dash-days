from dash import Dash
import dash_bootstrap_components as dbc
from layout.layout import app_layout
from callbacks.callbacks import register_callbacks

app = Dash('RL PoC', external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = app_layout

register_callbacks(app)

if __name__ == '__main__':
    app.run_server(debug=True, port=5000, use_reloader=True)
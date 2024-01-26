from dash import Dash
from layout.layout import app_layout

app = Dash('RL PoC',
           external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = app_layout

if __name__ == '__main__':
    app.run(debug=True, port=5000)